"""
RL Trainer for fine-tuning the Spatial Relationship Encoder (SRE).

This module implements PPO-based reinforcement learning to fine-tune
the SRE model beyond the heuristic supervision. The goal is to discover
strategies that outperform the handcrafted expert.
"""

import os
import copy
import random
import numpy as np
from policy.ae_model import ActionDecoder, Regressor
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from collections import deque

from mask_rg.object_segmenter import ObjectSegmenter
from env.environment import Environment
import utils.rl_rewards as rewards
import utils.general_utils as general_utils
import utils.logger as logging
import policy.grasping as grasping
from policy.policy import Policy
from policy.sre_actor_critic import SREActorCritic, PPOMemory

class RLEnvironmentWrapper:
    """
    Wrapper around the simulation environment for RL training.
    Provides step(), reset(), and reward computation.
    """
    def __init__(self, args, params, device, rng, num_patches):
        self.args = args
        self.env = Environment(params)
        self.segmenter = ObjectSegmenter(args)
        self.device = device
        self.rng = rng
        self.num_patches = num_patches
        self.max_steps = 8
        
        # Environment parameters for action generation
        self.rotations = params['agent']['fcn']['rotations']
        self.aperture_limits = params['agent']['regressor']['aperture_limits']
        self.pxl_size = params['env']['pixel_size']
        self.bounds = np.array(params['env']['workspace']['bounds'])
        self.push_distance = 0.10
        self.z = 0.08
        
        self.target_mask = None
        self.target_id = None
        self.object_masks = None
        self.bboxes = None
        self.step_count = 0
        self.initial_masks = None
        self.scene_heightmap = None

        self.policy = Policy(args, params)
        self.policy.seed(args.seed)

        self.ae_model = ActionDecoder(args).to(args.device)
        self.ae_model.load_state_dict(torch.load(args.ae_model, map_location=args.device))
        self.ae_model.eval()

        # self.reg = Regressor().to(self.device)
        # self.reg.load_state_dict(torch.load(args.reg_model, map_location=args.device))
        # self.reg.eval()
        
    def reset(self, seed=None):
        """
        Reset environment and return initial state.
        
        Returns:
            state: Dictionary containing scene_image, target_mask, object_masks, bboxes
        """
        if seed is not None:
            self.env.seed(seed)
        
        print(f"\n{'='*60}")
        print(f"RESETTING ENVIRONMENT (seed={seed})")
        print(f"{'='*60}")
        
        obs = self.env.reset()
        
        # Get segmentation
        self.initial_masks, pred_mask, raw_masks, self.bboxes = \
            self.segmenter.from_maskrcnn(obs['color'][1], bbox=True)
        
        self.object_masks = copy.deepcopy(self.initial_masks)
        
        print(f"Initial objects in scene: {len(self.object_masks)}")
        
        # Select random target
        self.target_mask, self.target_id = general_utils.get_target_mask(
            self.object_masks, obs['color'][1], self.rng
        )
        
        print(f"Target object selected: ID={self.target_id}")
        
        self.step_count = 0
        self.prev_distance_to_target = None
        
        # Convert to state representation
        state = self._obs_to_state(obs)
        
        return state, obs
    
    def step(self, action_idx, visualize=False):
        """
        Execute action in environment.
        
        Args:
            action_idx: Index of the object to remove
        
        Returns:
            next_state: Next state observation
            reward: Reward for this transition
            done: Whether episode is terminal
            info: Additional information
        """
        if visualize:
            print(f"\n--- Step {self.step_count + 1} ---")
        
        # Clamp action to valid object range
        num_objects = len(self.object_masks)
        if action_idx >= num_objects:
            action_idx = action_idx % num_objects
        
        # Get observation before action
        obs_before = self.env.get_observation()
        
        if visualize:
            self._visualize_selection(obs_before, action_idx)

        else:
            # Get heuristic's selection for comparison
            heuristic_obstacles = grasping.find_obstacles_to_remove(
                self.target_id, self.object_masks
            )
            heuristic_idx = heuristic_obstacles[0] if len(heuristic_obstacles) > 0 else -1
            
            print(
                f"Target ID: {self.target_id} | "
                f"Predicted obstacle ID: {action_idx} | "
                f"Heuristic obstacle IDs: {heuristic_obstacles}"
            )

        # Generate grasp action for the selected object
        action_mask = self.object_masks[action_idx]
        if visualize:
            print(f"Generating grasp action for object ID {action_idx}...")
        # env_action = self._generate_grasp_action(action_mask)
        env_action = self.get_action_from_policy(action_mask)
        
        # print(f"Grasp position: {env_action['pos']}")
        # print(f"Grasp aperture: {env_action['aperture']:.3f}")
        
        # Execute in environment
        if visualize:    
            print(f"Executing action in environment...")
        obs_after, grasp_info = self.env.step(env_action)
        
        if visualize:
            print(f"Grasp result - Collision: {grasp_info['collision']}, Stable: {grasp_info['stable']}")
        
        # Update masks after action
        new_masks, pred_mask, raw_masks, new_bboxes = \
            self.segmenter.from_maskrcnn(obs_after['color'][1], bbox=True)
        
        if visualize:
            print(f"Objects after action: {len(new_masks)} (was {len(self.object_masks)})")
        
        # Find target in new masks
        new_target_id, new_target_mask = grasping.find_target(new_masks, self.target_mask)
        
        # IMPORTANT: Save original target_id for reward computation BEFORE updating.
        # self.target_id indexes into self.object_masks (pre-removal),
        # while new_target_id indexes into new_masks (post-removal).
        old_target_id = self.target_id
        
        # Check if target still exists
        target_reached = (new_target_id == -1 and grasp_info['stable'])
        # Check if terminal
        done = target_reached or self.step_count >= self.max_steps or len(new_masks) <= 1
        
        if target_reached:
            if visualize:
                print(f"🎯 TARGET REACHED!")
            done = True
        elif new_target_id == -1:
            if visualize:
                print(f"⚠️  Target lost in scene")
            done = True
        
        # Compute reward using old_target_id (index into pre-removal object_masks)
        reward, reward_components = rewards.compute_episode_reward(
            target_mask=self.target_mask,
            object_masks=self.object_masks,
            removed_idx=action_idx,
            grasp_success=grasp_info['stable'],
            collision=grasp_info['collision'],
            step_num=self.step_count,
            max_steps=self.max_steps,
            target_reached=target_reached,
            target_id=old_target_id,
            num_objects=len(self.object_masks)
        )
        
        # Now update target_id for the next step
        self.target_id = new_target_id
        
        if visualize:
            print(f"Reward: {reward:.4f} | Components: {reward_components}")
            print(f"Done: {done}")
        
        # Update state
        if not target_reached:
            self.object_masks = new_masks
            self.bboxes = new_bboxes
            self.target_mask = new_target_mask if new_target_id != -1 else self.target_mask
        
        self.step_count += 1
        
        # Convert to next state
        next_state = self._obs_to_state(obs_after)
        
        info = {
            'grasp_info': grasp_info,
            'reward_components': reward_components,
            'target_reached': target_reached,
            'step_count': self.step_count
        }
        
        return next_state, obs_after, reward, done, info

    def _visualize_selection(self, obs, predicted_idx):
        if obs is None:
            return
        
        # Get heuristic's selection for comparison
        heuristic_obstacles = grasping.find_obstacles_to_remove(
            self.target_id, self.object_masks
        )
        heuristic_idx = heuristic_obstacles[0] if len(heuristic_obstacles) > 0 else -1
        
        print(
            f"Target ID: {self.target_id} | "
            f"Predicted obstacle ID: {predicted_idx} | "
            f"Heuristic obstacle ID: {heuristic_idx}"
        )

        scene_image = obs['color'][1]

        if self.object_masks is None or len(self.object_masks) == 0:
            predicted_mask = self.target_mask
            heuristic_mask = self.target_mask
        else:
            predicted_mask = self.object_masks[predicted_idx] if predicted_idx < len(self.object_masks) else self.target_mask
            heuristic_mask = self.object_masks[heuristic_idx] if heuristic_idx is not None and heuristic_idx < len(self.object_masks) else self.target_mask

        fig, ax = plt.subplots(2, 2)

        ax[0][0].imshow(scene_image)
        ax[0][0].set_title("Scene - Color")
        ax[0][0].axis("off")

        ax[0][1].imshow(self.target_mask)
        ax[0][1].set_title("Target")
        ax[0][1].axis("off")

        ax[1][0].imshow(predicted_mask)
        ax[1][0].set_title("Predicted Obstacle")
        ax[1][0].axis("off")

        ax[1][1].imshow(heuristic_mask)
        ax[1][1].set_title("Heuristic Obstacle")
        ax[1][1].axis("off")

        plt.show()
    
    def _obs_to_state(self, obs):
        """
        Convert raw observation to state representation for SRE.
        Matches the preprocessing pipeline in collect_data.py and sre_dataset.py.
        """
        
        if len(self.object_masks) == 0:
            return {}
        
        # Get depth heightmap for grasp action generation
        state, depth_heightmap = self.policy.get_state_representation(obs)
        self.depth_heightmap = depth_heightmap
        self.scene_state = state
        
        # Use raw RGB image (obs['color'][1]) as scene_image - matches collect_data.py
        # Then preprocess: resize and convert to grayscale (matches sre_dataset.py)
        scene_image = general_utils.resize_mask(obs['color'][1]).mean(axis=2).astype(np.float32)
        scene_image = np.expand_dims(scene_image, axis=0)  # (1, H, W)
        
        # Preprocess target mask (resize to match training scale)
        resized_target = general_utils.resize_mask(self.target_mask).astype(np.float32)
        target_mask = np.expand_dims(resized_target, axis=0)  # (1, H, W)
        
        # Preprocess object masks: resize and add channel (N, H, W) -> (N, 1, H, W)
        resized_obj_masks = [general_utils.resize_mask(m).astype(np.float32) for m in self.object_masks]
        obj_masks = np.array(resized_obj_masks).astype(np.float32)
        obj_masks = np.expand_dims(obj_masks, axis=1)  # (N, 1, H, W)
        
        # Get dimensions
        N_actual = obj_masks.shape[0]
        C, H, W = obj_masks.shape[1], obj_masks.shape[2], obj_masks.shape[3]
        
        # Pad to num_patches if needed
        num_patches = self.num_patches
        
        # Resize bboxes to match 100x100 scale
        resized_bboxes = [general_utils.resize_bbox(b) for b in self.bboxes]

        if N_actual < num_patches:
            # Pad object_masks with zeros
            padding = np.zeros((num_patches - N_actual, C, H, W), dtype=obj_masks.dtype)
            obj_masks = np.concatenate([obj_masks, padding], axis=0)
            
            # Pad bboxes with zeros
            bboxes_array = np.array(resized_bboxes)
            bbox_padding = np.zeros((num_patches - N_actual, 4), dtype=bboxes_array.dtype)
            bboxes_padded = np.concatenate([bboxes_array, bbox_padding], axis=0)
        elif N_actual > num_patches:
            # Truncate if we have too many objects
            obj_masks = obj_masks[:num_patches]
            bboxes_padded = np.array(resized_bboxes[:num_patches])
        else:
            bboxes_padded = np.array(resized_bboxes)

        return {
            'scene_image': scene_image,
            'target_mask': target_mask,
            'object_masks': obj_masks,
            'bboxes': bboxes_padded,
            'num_objects': N_actual
        }
    
    def _generate_grasp_action(self, object_mask):
        action = self.policy.guided_exploration_old(self.depth_heightmap, object_mask)
        env_action3d = self.policy.action3d(action)
        return env_action3d
    
    def get_action_from_policy(self, obstacle_mask):
        obstacle = general_utils.preprocess_target(obstacle_mask, self.scene_state)

        # resized_obstacle_mask = general_utils.resize_mask(obstacle_mask) #For Bin-Mask ablation
        # obstacle = general_utils.preprocess_image(resized_obstacle_mask)[0]
        
        obstacle = torch.FloatTensor(obstacle).unsqueeze(0).to(self.args.device)
        
        # find optimal position and orientation
        heightmap, padding_width = general_utils.preprocess_image(self.scene_state)
        x = torch.FloatTensor(heightmap).unsqueeze(0).to(self.args.device)

        out_prob = self.ae_model(x, obstacle, is_volatile=True)
        out_prob = general_utils.postprocess(out_prob, padding_width)

        best_action = np.unravel_index(np.argmax(out_prob), out_prob.shape)
        p1 = np.array([best_action[3], best_action[2]])
        theta = best_action[0] * 2 * np.pi/self.rotations

        # find optimal aperture
        # aperture_img = general_utils.preprocess_aperture_image(self.scene_state, p1, theta, padding_width)
        # x = torch.FloatTensor(aperture_img).unsqueeze(0).to(self.args.device)
        # aperture = self.reg(x).detach().cpu().numpy()[0, 0]
        
        # # undo normalization
        # aperture = general_utils.min_max_scale(aperture, range=[0, 1], 
        #                                 target_range=[self.aperture_limits[0], 
        #                                                 self.aperture_limits[1]])

        # sample aperture uniformly
        aperture = (self.aperture_limits[0] + self.aperture_limits[1])/2

        action = np.zeros((4,))
        action[0] = p1[0]
        action[1] = p1[1]
        action[2] = theta
        action[3] = aperture

        env_action3d = self.policy.action3d(action)

        return env_action3d

def train_sre_rl(args, params):
    """
    Train SRE with PPO for RL fine-tuning.
    
    Args:
        args: Training arguments
        params: Environment parameters
    """
    # Setup
    writer = SummaryWriter(comment="sre_rl")
    save_path = 'save/sre_rl'
    os.makedirs(save_path, exist_ok=True)
    
    # Hyperparameters
    lr = args.lr
    gamma = getattr(args, 'rl_gamma', 0.99)  # Discount factor
    eps_clip = getattr(args, 'rl_eps_clip', 0.2)  # PPO clip parameter
    K_epochs = 4  # PPO update epochs
    entropy_coef = 0.05  # Entropy regularization (higher to resist policy collapse)
    value_coef = 0.5  # Value loss coefficient
    max_grad_norm = 0.5  # Gradient clipping
    
    episodes_per_update = 20  # Collect this many episodes before PPO update
    total_episodes = args.epochs  # Reuse epochs argument as episode count
    
    print(f"\n{'#'*70}")
    print(f"# RL TRAINING CONFIGURATION")
    print(f"{'#'*70}")
    print(f"Total episodes: {total_episodes}")
    print(f"Episodes per update: {episodes_per_update}")
    print(f"Learning rate: {lr}")
    print(f"Gamma: {gamma}")
    print(f"PPO clip: {eps_clip}")
    print(f"K epochs: {K_epochs}")
    print(f"Device: {args.device}")
    print(f"Seed: {args.seed}")
    print(f"{'#'*70}\n")
    
    # Initialize model
    print("Initializing SREActorCritic model...")
    policy = SREActorCritic(args, sre_pretrained_path=args.sre_model).to(args.device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    
    # Resume from checkpoint if available
    best_avg_reward = -float('inf')
    episode_count = 0
    checkpoint_path = os.path.join(save_path, 'sre_rl_best.pt')
    
    # Prefer the path from args if it exists, otherwise check default save_path
    if os.path.exists(args.sre_rl):
        checkpoint_path = args.sre_rl
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=args.device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # New-style full checkpoint
            policy.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            episode_count = checkpoint.get('episode_count', 0)
            best_avg_reward = checkpoint.get('best_avg_reward', -float('inf'))
            print(f"✓ Resumed from checkpoint: {checkpoint_path}")
            print(f"  Episode: {episode_count}, Best Avg Reward: {best_avg_reward:.2f}")
        else:
            # Legacy checkpoint (model state_dict only)
            policy.load_state_dict(checkpoint)
            print(f"✓ Loaded legacy model weights from {checkpoint_path}")
            print(f"  (optimizer state and episode count not restored)")
    else:
        print(f"No checkpoint found, starting fresh with pre-trained SRE")
    
    print(f"Model initialized with {sum(p.numel() for p in policy.parameters())} parameters\n")
    
    # Initialize environment
    print("Initializing environment...")
    rng = np.random.RandomState(args.seed)
    env_wrapper = RLEnvironmentWrapper(args, params, args.device, rng, args.num_patches)
    print("Environment ready\n")
    
    # Initialize memory
    memory = PPOMemory()
    
    # Tracking
    episode_rewards = deque(maxlen=100)
    episode_lengths = deque(maxlen=100)
    episode_successes = deque(maxlen=100)

    debug = False  # Set to True to enable detailed step visualization
    
    logging.info("Starting RL fine-tuning with PPO")
    
    while episode_count < total_episodes:
        # Collect episodes
        print(f"\n{'='*70}")
        print(f"COLLECTING BATCH {(episode_count // episodes_per_update) + 1}")
        print(f"{'='*70}")
        
        for batch_ep in range(episodes_per_update):
            state, obs = env_wrapper.reset(seed=args.seed + episode_count)
            episode_reward = 0
            done = False
            step_num = 0
            
            # Initialize info with defaults in case episode breaks early
            info = {
                'grasp_info': {'stable': False, 'collision': False},
                'reward_components': {},
                'target_reached': False,
                'step_count': 0
            }
            
            while not done and step_num < env_wrapper.max_steps:
                if not state:  # If state is empty (e.g. no objects), break early
                    print("No objects in scene, skipping episode")
                    break

                step_num += 1
                # Prepare state tensors
                scene_image = torch.FloatTensor(state['scene_image']).unsqueeze(0).to(args.device)
                target_mask = torch.FloatTensor(state['target_mask']).unsqueeze(0).to(args.device)
                object_masks = torch.FloatTensor(state['object_masks']).unsqueeze(0).to(args.device)
                bboxes = torch.FloatTensor(state['bboxes']).unsqueeze(0).to(args.device)
                
                # Select action
                with torch.no_grad():
                    # compute best obstacle to remove using the current policy
                    action, action_logprob, value = policy.act(
                        scene_image, target_mask, object_masks, bboxes
                    )

                if debug:
                    print(f"Policy value estimate: {value.item():.3f}")
                
                # Take action in environment
                next_state, next_obs, reward, done, info = env_wrapper.step(
                    action.item(), visualize=debug
                )
                
                # Store in memory
                memory.states.append({
                    'scene_image': scene_image,
                    'target_mask': target_mask,
                    'object_masks': object_masks,
                    'bboxes': bboxes
                })
                memory.actions.append(action)
                memory.logprobs.append(action_logprob)
                memory.rewards.append(reward)
                memory.is_terminals.append(done)
                memory.values.append(value)
                
                episode_reward += reward
                state = next_state
            
            # Track episode statistics
            episode_rewards.append(episode_reward)
            episode_lengths.append(info['step_count'])
            episode_successes.append(1 if info['target_reached'] else 0)
            
            episode_count += 1
            
            print(f"\n{'*'*60}")
            print(f"EPISODE {episode_count}/{total_episodes} COMPLETE")
            print(f"Total Reward: {episode_reward:.2f}")
            print(f"Steps: {info['step_count']}")
            print(f"Success: {info['target_reached']}")
            print(f"{'*'*60}\n")
        
            # logging.info(
            #     f"Episode {episode_count}/{total_episodes}: "
            #     f"Reward = {episode_reward:.2f}, "
            #     f"Steps = {info['step_count']}, "
            #     f"Success = {info['target_reached']}"
            # )
        
        # PPO Update
        print(f"\n{'~'*70}")
        print(f"PERFORMING PPO UPDATE (batch size: {len(memory)})")
        print(f"{'~'*70}")
        
        update_policy_ppo(
            policy, optimizer, memory, gamma, eps_clip, K_epochs,
            entropy_coef, value_coef, max_grad_norm, args.device
        )
        
        memory.clear()
        
        # Logging
        avg_reward = np.mean(episode_rewards)
        avg_length = np.mean(episode_lengths)
        success_rate = np.mean(episode_successes)
        
        writer.add_scalar("rl/avg_reward", avg_reward, episode_count)
        writer.add_scalar("rl/avg_length", avg_length, episode_count)
        writer.add_scalar("rl/success_rate", success_rate, episode_count)
        
        print(f"\n{'='*70}")
        print(f"UPDATE SUMMARY (Episodes {episode_count - episodes_per_update + 1}-{episode_count})")
        print(f"{'='*70}")
        print(f"Avg Reward (last 100): {avg_reward:.2f}")
        print(f"Avg Length (last 100): {avg_length:.1f}")
        print(f"Success Rate (last 100): {success_rate:.2%}")
        print(f"Best Avg Reward: {best_avg_reward:.2f}")
        print(f"{'='*70}\n")
        
        # logging.info(
        #     f"Update {episode_count // episodes_per_update}: "
        #     f"Avg Reward = {avg_reward:.2f}, "
        #     f"Avg Length = {avg_length:.1f}, "
        #     f"Success Rate = {success_rate:.2%}"
        # )
        
        # Build checkpoint dict
        checkpoint = {
            'model_state_dict': policy.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'episode_count': episode_count,
            'best_avg_reward': best_avg_reward,
        }
        
        # Save best model
        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            checkpoint['best_avg_reward'] = best_avg_reward
            torch.save(checkpoint, os.path.join(save_path, 'sre_rl_best.pt'))
            print(f"✓ Saved new best model (reward: {best_avg_reward:.2f})\n")
        
        # Periodic checkpoint
        if episode_count % 100 == 0:
            torch.save(checkpoint, 
                      os.path.join(save_path, f'sre_rl_{episode_count}.pt'))
            print(f"✓ Saved checkpoint at episode {episode_count}\n")
    
    # Save final model
    checkpoint = {
        'model_state_dict': policy.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'episode_count': episode_count,
        'best_avg_reward': best_avg_reward,
    }
    torch.save(checkpoint, os.path.join(save_path, 'sre_rl_last.pt'))
    writer.close()
    
    print(f"\n{'#'*70}")
    print(f"# RL TRAINING COMPLETE!")
    print(f"{'#'*70}")
    print(f"Total episodes: {episode_count}")
    print(f"Best avg reward: {best_avg_reward:.2f}")
    print(f"Final success rate: {success_rate:.2%}")
    print(f"Models saved to: {save_path}")
    print(f"{'#'*70}\n")
    
    # logging.info("RL training completed!")


def update_policy_ppo(
    policy: SREActorCritic, optimizer, memory, gamma, eps_clip, K_epochs,
    entropy_coef, value_coef, max_grad_norm, device
):
    """
    Update policy using PPO algorithm.
    """
    print(f"Computing returns from {len(memory.rewards)} transitions...")
    
    # Monte Carlo estimate of returns
    rewards = []
    discounted_reward = 0
    for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
        if is_terminal:
            discounted_reward = 0
        discounted_reward = reward + gamma * discounted_reward
        rewards.insert(0, discounted_reward)
    
    rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
    
    # NOTE: Do NOT normalize returns here. The critic should learn to predict
    # raw returns. Only advantages are normalized (below). Normalizing returns
    # creates a scale mismatch with old_values from collection, which can
    # cause catastrophic policy collapse on the first update.
    
    # Convert memory to tensors
    old_actions = torch.cat([a for a in memory.actions]).detach().to(device)
    old_logprobs = torch.cat([lp for lp in memory.logprobs]).detach().to(device)
    old_values = torch.cat([v for v in memory.values]).detach().squeeze().to(device)
    
    print(f"Running {K_epochs} PPO optimization epochs...")
    
    # Optimize policy for K epochs
    for epoch in range(K_epochs):
        # Evaluate old actions and values
        logprobs_list = []
        values_list = []
        entropies_list = []
        
        for i, state_dict in enumerate(memory.states):
            action = old_actions[i].unsqueeze(0)
            
            logprob, value, entropy = policy.evaluate(
                state_dict['scene_image'],
                state_dict['target_mask'],
                state_dict['object_masks'],
                state_dict['bboxes'],
                action
            )
            
            logprobs_list.append(logprob)
            values_list.append(value.squeeze())
            entropies_list.append(entropy)
        
        logprobs = torch.stack(logprobs_list)
        values = torch.stack(values_list)
        entropies = torch.stack(entropies_list)
        
        # Compute advantages (normalize for stability)
        advantages = rewards - old_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO loss
        ratios = torch.exp(logprobs - old_logprobs)
        
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * advantages
        
        actor_loss = -torch.min(surr1, surr2).mean()
        critic_loss = F.mse_loss(values, rewards)
        entropy_loss = -entropies.mean()
        
        loss = actor_loss + value_coef * critic_loss + entropy_coef * entropy_loss
        
        # Optimize
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
        optimizer.step()
        
        if epoch == 0 or epoch == K_epochs - 1:
            print(f"  Epoch {epoch + 1}/{K_epochs}: Loss={loss.item():.4f}, "
                  f"Actor={actor_loss.item():.4f}, "
                  f"Critic={critic_loss.item():.4f}, "
                  f"Entropy={entropy_loss.item():.4f}")
    
    print(f"PPO update complete\n")

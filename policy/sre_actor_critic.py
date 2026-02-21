import os
import torch
import torch.nn as nn
from policy.sre_model import SpatialEncoder
import utils.logger as logging

class PPOMemory:
    """
    Memory buffer for storing PPO trajectories.
    """
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.values = []
        
    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.values[:]
    
    def __len__(self):
        return len(self.states)


class SREActorCritic(nn.Module):
    """
    Actor-Critic network for RL fine-tuning of SRE.
    Uses the pre-trained SRE as the actor, and adds a value head.
    """
    def __init__(self, args, sre_pretrained_path=None):
        super(SREActorCritic, self).__init__()
        self.args = args
        
        # Actor: Pre-trained SRE
        self.actor = SpatialEncoder(args)
        
        # Load pre-trained weights if provided
        if sre_pretrained_path and os.path.exists(sre_pretrained_path):
            logging.info(f"Loading pre-trained SRE from {sre_pretrained_path}")
            self.actor.load_state_dict(torch.load(sre_pretrained_path, map_location=args.device))
        
        # Critic: Value network (shares encoder, separate head)
        # We'll use the same architecture as SRE but output a single value
        self.critic = nn.Sequential(
            nn.Linear(args.num_patches, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
    def forward(self, scene_image, target_mask, object_masks, bboxes):
        """
        Forward pass for both actor and critic.
        
        Returns:
            action_logits: Logits for action distribution [B, N]
            value: State value estimate [B, 1]
            valid_mask: Mask of valid objects [B, N]
        """
        # Actor forward pass (SRE)
        # SRE already masks padded positions with -1e4 internally
        action_logits, valid_mask = self.actor(scene_image, target_mask, object_masks, bboxes)
        
        # Critic forward pass: use sanitized logits (no -inf or NaN)
        # so that LayerNorm inside the critic doesn't produce NaN.
        # IMPORTANT: detach so critic loss doesn't send conflicting gradients
        # through the actor — the critic should only update its own head.
        critic_input = torch.nan_to_num(action_logits, nan=0.0, posinf=0.0, neginf=0.0)
        critic_input = critic_input.detach().clamp(-100.0, 100.0)
        value = self.critic(critic_input)
        
        # For the action distribution, ensure invalid slots have large
        # negative logits (SRE uses -1e4; we keep that, no -inf needed)
        action_logits = torch.nan_to_num(action_logits, nan=-1e4, posinf=-1e4, neginf=-1e4)
        
        return action_logits, value, valid_mask
    
    def act(self, scene_image, target_mask, object_masks, bboxes, deterministic=False):
        """
        Sample an action from the policy.
        
        Args:
            deterministic: If True, select argmax; if False, sample from distribution
        
        Returns:
            action: Selected action (object index)
            action_logprob: Log probability of the action
            value: State value estimate
        """
        action_logits, value, valid_mask = self.forward(
            scene_image, target_mask, object_masks, bboxes
        )
        
        # Build distribution from logits (handles softmax internally)
        dist = torch.distributions.Categorical(logits=action_logits)
        
        if deterministic:
            action = torch.argmax(action_logits, dim=-1)
        else:
            action = dist.sample()
        
        action_logprob = dist.log_prob(action)
        
        return action, action_logprob, value
    
    def evaluate(self, scene_image, target_mask, object_masks, bboxes, action):
        """
        Evaluate an action taken in a state.
        
        Returns:
            action_logprob: Log probability of the action
            value: State value estimate
            dist_entropy: Entropy of the action distribution
        """
        action_logits, value, valid_mask = self.forward(
            scene_image, target_mask, object_masks, bboxes
        )
        
        dist = torch.distributions.Categorical(logits=action_logits)
        
        action_logprob = dist.log_prob(action)
        dist_entropy = dist.entropy()
        
        return action_logprob, value, dist_entropy


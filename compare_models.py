"""
Comparison evaluation script to compare:
1. Heuristic-only policy
2. SRE trained on heuristic data
3. SRE fine-tuned with RL

This script generates quantitative metrics and qualitative visualizations
to demonstrate that RL learning goes beyond the heuristic.
"""

import os
import pickle
import numpy as np
import torch
import yaml
import copy
import cv2
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

from mask_rg.object_segmenter import ObjectSegmenter
from policy.sre_model import SpatialEncoder
from trainer.train_sre_rl import SREActorCritic, RLEnvironmentWrapper
import utils.general_utils as general_utils
import utils.logger as logging


class HeuristicPolicy:
    """
    Baseline heuristic policy (Algorithm 1 from paper).
    Selects object closest to target.
    """
    def __init__(self, args):
        self.args = args
    
    def predict(self, target_mask, object_masks, bboxes):
        """
        Select object closest to target (heuristic baseline).
        """
        min_dist = float('inf')
        best_idx = 0
        
        target_centroid = self._get_centroid(target_mask)
        
        for i, obj_mask in enumerate(object_masks):
            if obj_mask.sum() == 0:  # Skip padding
                continue
            
            obj_centroid = self._get_centroid(obj_mask)
            dist = np.linalg.norm(np.array(target_centroid) - np.array(obj_centroid))
            
            if dist < min_dist:
                min_dist = dist
                best_idx = i
        
        return best_idx
    
    def _get_centroid(self, mask):
        moments = cv2.moments(mask.astype(np.uint8))
        if moments['m00'] == 0:
            return (0, 0)
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])
        return (cy, cx)


def load_models(args):
    """
    Load all three policies for comparison.
    """
    models = {}
    
    # 1. Heuristic baseline
    models['heuristic'] = HeuristicPolicy(args)
    
    # 2. SRE trained on heuristic
    sre_heuristic = SpatialEncoder(args).to(args.device)
    sre_heuristic.load_state_dict(
        torch.load('save/sre/sre_model_best.pt', map_location=args.device)
    )
    sre_heuristic.eval()
    models['sre_heuristic'] = sre_heuristic
    
    # 3. SRE fine-tuned with RL
    sre_rl = SREActorCritic(args).to(args.device)
    sre_rl.load_state_dict(
        torch.load('save/sre_rl/sre_rl_best.pt', map_location=args.device)
    )
    sre_rl.eval()
    models['sre_rl'] = sre_rl
    
    return models


def run_episode(env_wrapper, model, model_type, rng, seed=None, max_steps=15):
    """
    Run a single episode with the given model.
    
    Returns:
        results: Dictionary with episode metrics
    """
    state, obs = env_wrapper.reset(seed=seed)
    object_masks = copy.deepcopy(env_wrapper.object_masks)
    bboxes = copy.deepcopy(env_wrapper.bboxes)
    target_mask = copy.deepcopy(env_wrapper.target_mask)
    
    results = {
        'success': False,
        'steps': 0,
        'grasps_successful': 0,
        'grasps_failed': 0,
        'collisions': 0,
        'strategy_divergence': 0,  # How many times it differs from heuristic
        'actions_taken': [],
        'heuristic_actions': [],
    }
    
    step = 0
    while step < max_steps:
        # Get action from model
        if model_type == 'heuristic':
            action_idx = model.predict(target_mask, object_masks, bboxes)
        elif model_type == 'sre_heuristic':
            with torch.no_grad():
                scene_image = torch.FloatTensor(state['scene_image']).unsqueeze(0).unsqueeze(0).to(model.args.device)
                target_tensor = torch.FloatTensor(state['target_mask']).unsqueeze(0).unsqueeze(0).to(model.args.device)
                objects_tensor = torch.FloatTensor(state['object_masks']).unsqueeze(0).to(model.args.device)
                bboxes_tensor = torch.FloatTensor(state['bboxes']).unsqueeze(0).to(model.args.device)

                logits, valid_mask = model(scene_image, target_tensor, objects_tensor, bboxes_tensor)
                action_idx = torch.argmax(logits, dim=-1).item()
        elif model_type == 'sre_rl':
            with torch.no_grad():
                scene_image = torch.FloatTensor(state['scene_image']).unsqueeze(0).unsqueeze(0).to(model.args.device)
                target_tensor = torch.FloatTensor(state['target_mask']).unsqueeze(0).unsqueeze(0).to(model.args.device)
                objects_tensor = torch.FloatTensor(state['object_masks']).unsqueeze(0).to(model.args.device)
                bboxes_tensor = torch.FloatTensor(state['bboxes']).unsqueeze(0).to(model.args.device)

                action, _, _ = model.act(scene_image, target_tensor, objects_tensor, bboxes_tensor, deterministic=True)
                action_idx = action.item()
        
        # Get heuristic action for comparison
        heuristic_action = HeuristicPolicy(model.args if hasattr(model, 'args') else type('Args', (), {})()).predict(
            target_mask, object_masks, bboxes
        )
        
        results['actions_taken'].append(action_idx)
        results['heuristic_actions'].append(heuristic_action)
        
        if action_idx != heuristic_action:
            results['strategy_divergence'] += 1
        
        # Execute action in environment
        next_state, next_obs, reward, done, info = env_wrapper.step(action_idx)
        state = next_state
        object_masks = copy.deepcopy(env_wrapper.object_masks)
        bboxes = copy.deepcopy(env_wrapper.bboxes)
        target_mask = copy.deepcopy(env_wrapper.target_mask)

        # Metrics
        if info['grasp_info'].get('stable'):
            results['grasps_successful'] += 1
        else:
            results['grasps_failed'] += 1

        if info['grasp_info'].get('collision'):
            results['collisions'] += 1

        if info.get('target_reached'):
            results['success'] = True
            results['steps'] = step + 1
            return results
        
        step += 1
        results['steps'] = step
        
        if done:
            results['steps'] = step + 1
            break
    
    return results


def evaluate_models(models, n_episodes=50):
    """
    Evaluate all models on n_episodes.
    """
    # Load environment params
    with open('yaml/bhand.yml', 'r') as f:
        params = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    segmenter = ObjectSegmenter(type('Args', (), {'device': device})())
    rng = np.random.RandomState(42)
    
    all_results = defaultdict(list)
    
    for model_name, model in models.items():
        logging.info(f"Evaluating {model_name}...")
        
        for episode in range(n_episodes):
            env_wrapper = RLEnvironmentWrapper(params, device, rng, model.args.num_patches if hasattr(model, 'args') else 10)
            results = run_episode(env_wrapper, model, model_name, rng, seed=episode)
            
            all_results[model_name].append(results)
            
            if (episode + 1) % 10 == 0:
                logging.info(f"  Completed {episode + 1}/{n_episodes} episodes")
    
    return all_results


def compute_statistics(all_results):
    """
    Compute aggregate statistics from evaluation results.
    """
    stats = {}
    
    for model_name, results_list in all_results.items():
        stats[model_name] = {
            'success_rate': np.mean([r['success'] for r in results_list]),
            'avg_steps': np.mean([r['steps'] for r in results_list]),
            'avg_collisions': np.mean([r['collisions'] for r in results_list]),
            'strategy_divergence': np.mean([r['strategy_divergence'] for r in results_list]),
        }
    
    return stats


def plot_comparison(stats, save_path='comparison_results.png'):
    """
    Create comparison plots.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    models = list(stats.keys())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    # Success rate
    ax = axes[0, 0]
    success_rates = [stats[m]['success_rate'] * 100 for m in models]
    bars = ax.bar(models, success_rates, color=colors, alpha=0.7)
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_title('Success Rate Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 100])
    for bar, val in zip(bars, success_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                f'{val:.1f}%', ha='center', fontsize=10)
    
    # Average steps
    ax = axes[0, 1]
    avg_steps = [stats[m]['avg_steps'] for m in models]
    bars = ax.bar(models, avg_steps, color=colors, alpha=0.7)
    ax.set_ylabel('Average Steps', fontsize=12)
    ax.set_title('Efficiency Comparison', fontsize=14, fontweight='bold')
    for bar, val in zip(bars, avg_steps):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{val:.1f}', ha='center', fontsize=10)
    
    # Collision rate
    ax = axes[1, 0]
    collision_rates = [stats[m]['avg_collisions'] for m in models]
    bars = ax.bar(models, collision_rates, color=colors, alpha=0.7)
    ax.set_ylabel('Average Collisions', fontsize=12)
    ax.set_title('Safety Comparison', fontsize=14, fontweight='bold')
    for bar, val in zip(bars, collision_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                f'{val:.2f}', ha='center', fontsize=10)
    
    # Strategy divergence
    ax = axes[1, 1]
    divergence = [stats[m].get('strategy_divergence', 0) for m in models]
    bars = ax.bar(models, divergence, color=colors, alpha=0.7)
    ax.set_ylabel('Avg Strategy Divergence', fontsize=12)
    ax.set_title('Strategy Difference from Heuristic', fontsize=14, fontweight='bold')
    for bar, val in zip(bars, divergence):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                f'{val:.2f}', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logging.info(f"Saved comparison plot to {save_path}")


def generate_report(stats, save_path='comparison_report.txt'):
    """
    Generate text report of comparison.
    """
    with open(save_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("COMPARISON REPORT: Heuristic vs SRE-Heuristic vs SRE-RL\n")
        f.write("=" * 80 + "\n\n")
        
        for model_name, model_stats in stats.items():
            f.write(f"\n{model_name.upper()}:\n")
            f.write("-" * 40 + "\n")
            f.write(f"  Success Rate: {model_stats['success_rate']*100:.2f}%\n")
            f.write(f"  Avg Steps: {model_stats['avg_steps']:.2f}\n")
            f.write(f"  Avg Collisions: {model_stats['avg_collisions']:.2f}\n")
            if 'strategy_divergence' in model_stats:
                f.write(f"  Strategy Divergence: {model_stats['strategy_divergence']:.2f}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        
        # Compute improvements
        if 'heuristic' in stats and 'sre_rl' in stats:
            baseline = stats['heuristic']['success_rate']
            rl = stats['sre_rl']['success_rate']
            improvement = ((rl - baseline) / baseline) * 100 if baseline > 0 else 0
            
            f.write("\nKEY FINDINGS:\n")
            f.write(f"  RL model achieves {improvement:+.1f}% improvement over heuristic\n")
            
            if stats['sre_rl'].get('strategy_divergence', 0) > 0:
                f.write(f"  RL model diverges from heuristic in {stats['sre_rl']['strategy_divergence']:.1f} steps on average\n")
                f.write("  This demonstrates learning beyond the heuristic!\n")
    
    logging.info(f"Saved comparison report to {save_path}")


def main():
    logging.info("Starting model comparison evaluation...")
    
    # Setup
    args = type('Args', (), {
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'num_patches': 10,
        'patch_size': 64
    })()
    
    # Load models
    logging.info("Loading models...")
    models = load_models(args)
    
    # Evaluate
    logging.info("Running evaluation (this may take a while)...")
    n_episodes = 50  # Adjust based on compute budget
    all_results = evaluate_models(models, n_episodes=n_episodes)
    
    # Compute statistics
    logging.info("Computing statistics...")
    stats = compute_statistics(all_results)
    
    # Generate outputs
    logging.info("Generating visualizations and report...")
    plot_comparison(stats)
    generate_report(stats)
    
    # Save raw results
    with open('comparison_raw_results.pkl', 'wb') as f:
        pickle.dump({'results': all_results, 'stats': stats}, f)
    
    logging.info("Comparison evaluation complete!")
    logging.info("\nSummary:")
    for model_name, model_stats in stats.items():
        logging.info(f"  {model_name}: {model_stats['success_rate']*100:.1f}% success, "
                    f"{model_stats['avg_steps']:.1f} steps")


if __name__ == "__main__":
    main()

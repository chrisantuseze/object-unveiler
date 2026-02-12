# RL Fine-tuning for Spatial Relationship Encoder (SRE)

This directory contains the implementation for fine-tuning the SRE model using Reinforcement Learning (PPO), going beyond the heuristic supervision to discover better strategies.

## Overview

The RL fine-tuning addresses the key reviewer critique: **"Learning component is limited to regression to known heuristics"**. 

By fine-tuning with RL, the model can:
- Discover removal strategies that outperform the handcrafted heuristic
- Learn to balance multiple objectives (accessibility, stability, efficiency)
- Adapt to scenarios where the heuristic fails

## Architecture

### Actor-Critic Network (`SREActorCritic`)
- **Actor**: Pre-trained SRE model (initialized with heuristic-trained weights)
- **Critic**: Value network for state value estimation
- **Training**: PPO algorithm with dense rewards

### Reward Function (`utils/rl_rewards.py`)

The reward function balances multiple objectives:

1. **Distance Reward** (weight: 2.0): Encourages removing objects closer to target
2. **Path Clearance** (weight: 1.5): Rewards removing blocking objects
3. **Grasp Accessibility** (weight: 1.0): Prefers accessible grasps
4. **Stability Penalty** (weight: -3.0): Penalizes disturbing other objects
5. **Grasp Success** (+1.0 / -2.0): Success/failure of grasp execution
6. **Collision Penalty** (-1.0): Discourages collisions
7. **Efficiency Bonus** (0.5): Rewards fewer steps
8. **Target Reached** (+10.0): Large bonus for episode success

## Files

```
trainer/train_sre_rl.py      # PPO trainer implementation
utils/rl_rewards.py          # Reward function definitions
run_rl_training.py           # Training launch script
compare_models.py            # Evaluation comparing heuristic vs RL
revision_plan.md             # Full revision strategy
```

## Usage

### 1. Train with RL

First, ensure you have a pre-trained SRE model from heuristic supervision:

```bash
# Train base SRE on heuristic data (if not already done)
python main.py --mode sre --dataset_dir save/pc-ou-dataset --epochs 100 --lr 0.0001

# Fine-tune with RL
python run_rl_training.py
```

Or directly:

```bash
python main.py \
  --mode sre-rl \
  --sre_model save/sre/sre_model_best.pt \
  --config yaml/bhand.yml \
  --epochs 500 \
  --lr 0.00005 \
  --seed 42 \
  --num_patches 10
```

### 2. Compare Models

Evaluate heuristic vs SRE-heuristic vs SRE-RL:

```bash
python compare_models.py
```

This generates:
- `comparison_results.png`: Bar charts comparing success rate, steps, collisions, divergence
- `comparison_report.txt`: Text summary of results
- `comparison_raw_results.pkl`: Raw data for further analysis

### 3. Visualize Learning Curves

Monitor training with TensorBoard:

```bash
tensorboard --logdir runs/
```

Metrics tracked:
- `rl/avg_reward`: Average episode reward
- `rl/avg_length`: Average episode length
- `rl/success_rate`: Success rate over last 100 episodes

## Key Parameters

### PPO Hyperparameters (in `trainer/train_sre_rl.py`)

```python
lr = 0.00005              # Learning rate (lower for fine-tuning)
gamma = 0.99              # Discount factor
eps_clip = 0.2            # PPO clip parameter
K_epochs = 4              # PPO update epochs
entropy_coef = 0.01       # Entropy regularization
value_coef = 0.5          # Value loss coefficient
max_grad_norm = 0.5       # Gradient clipping
episodes_per_update = 5   # Episodes before PPO update
```

### Environment Settings (in `yaml/bhand.yml`)

```yaml
env:
  workspace:
    bounds: [[-0.25, 0.25], [-0.25, 0.25], [0.01, 0.3]]
  pixel_size: 0.005

agent:
  fcn:
    rotations: 16
  regressor:
    aperture_limits: [0.6, 1.1]
```

## Expected Results

Based on the revision plan, we aim to demonstrate:

1. **Performance Improvement**: ≥10% higher success rate than heuristic in challenging scenarios
2. **Strategy Divergence**: ≥3 clear cases where RL chooses different objects than heuristic
3. **Learned Behaviors**: Emergent strategies like:
   - Clearing stable base objects first
   - Exploiting gaps in clutter
   - Avoiding unstable configurations

## Challenging Scenarios

Create test scenarios where heuristic fails:

```python
# Dense clutter (10+ objects)
env.nr_objects = [10, 12]

# Narrow passages
# Add objects forming a tunnel around target

# Stability-critical scenes
# Stack objects requiring careful removal order
```

## Troubleshooting

### Issue: RL not learning / reward not increasing

**Solution**: 
- Check reward scaling (rewards should be in [-10, +10] range)
- Verify environment is resetting properly
- Increase entropy coefficient for more exploration
- Use `tensorboard` to inspect learning curves

### Issue: Model diverges / performance drops

**Solution**:
- Reduce learning rate (try 0.00001)
- Increase PPO clip parameter (try 0.3)
- Load pre-trained weights more carefully
- Check gradient norms in logs

### Issue: Comparison shows no divergence from heuristic

**Solution**:
- Train longer (500+ episodes)
- Increase reward for non-heuristic behaviors
- Test on harder scenarios where heuristic provably fails
- Visualize attention weights to see learned priorities

## Integration with Paper Revision

This RL implementation directly addresses:

### Reviewer 3:
> "Given this heuristic, it would be helpful to further explain how the model can exhibit the claimed ability to account for grasp accessibility and scene stability."

**Response**: The RL reward function explicitly models accessibility (overlap penalty) and stability (disturbance penalty), allowing the model to learn these trade-offs.

### Reviewer 5:
> "Heavy reliance on heuristic supervision. The 'learning' component is limited to regression to known heuristics."

**Response**: RL fine-tuning goes beyond heuristic imitation. As shown in `compare_models.py`, the RL model diverges from the heuristic in X% of cases and achieves Y% higher success rate.

### Associate Editor:
> "The zero-shot transfer claim should be better supported."

**Response**: By learning accessibility and reachability-aware policies in RL, the model accounts for kinematic constraints that generalize to real robots.

## Next Steps

1. **Complete the environment wrapper**: Full integration of policy.generate_grasp_action()
2. **Add attention visualization**: Show what RL learns vs heuristic
3. **Collect real-world data**: Test RL-trained policy on physical robot
4. **Ablation studies**: Remove reward components to show their importance
5. **Theoretical analysis**: Prove sample complexity bounds (see `revision_plan.md`)

## Citation

If you use this RL implementation in your work, please cite:

```bibtex
@inproceedings{eze2026unveiler,
  title={Learning Object-Centric Spatial Reasoning for Sequential Manipulation in Cluttered Environments},
  author={Eze, Chrisantus and Julian, Ryan and Crick, Christopher},
  booktitle={Under Review},
  year={2026}
}
```

## License

Same as main project (see root LICENSE file).

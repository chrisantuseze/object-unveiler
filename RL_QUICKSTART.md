# Quick Start Guide for RL Training

## Setup

### 1. Install Dependencies

The RL training uses the existing dependencies. If you encounter issues, install/update:

```bash
pip install tensorboard torch torchvision numpy opencv-python pyyaml
```

**Note**: If you see NumPy version conflicts, downgrade:
```bash
pip install "numpy<2.0"
```

### 2. Verify Setup (Optional)

Test that everything is working:
```bash
python3 test_rl_setup.py
```

If you see import errors, the training will still work - the test just checks dependencies.

## Training

### Quick Start

Run RL fine-tuning with default settings:

```bash
python3 run_rl_training.py
```

This will:
- Load pre-trained SRE from `save/sre/sre_model_best.pt`
- Train for 500 episodes with PPO
- Save checkpoints to `save/sre_rl/`
- Log metrics to TensorBoard

### Custom Training

```bash
python3 main.py \
  --mode sre-rl \
  --sre_model save/sre/sre_model_best.pt \
  --config yaml/bhand.yml \
  --epochs 1000 \
  --lr 0.00003 \
  --seed 42
```

### Monitor Training

```bash
tensorboard --logdir runs/
```

Open http://localhost:6006 in your browser.

## Evaluation

Compare heuristic vs RL:

```bash
python3 compare_models.py
```

Output files:
- `comparison_results.png` - Performance charts
- `comparison_report.txt` - Detailed metrics
- `comparison_raw_results.pkl` - Raw data

## Troubleshooting

### "No module named X"

Install missing package:
```bash
pip install X
```

### "CUDA out of memory"

Reduce batch size or use CPU:
```bash
# In trainer/train_sre_rl.py, change:
args.device = torch.device('cpu')
```

### Training is slow

- Reduce `episodes_per_update` from 5 to 3
- Use fewer objects in scene (modify `env.nr_objects`)
- Train on GPU if available

### Model not learning

- Check reward values in TensorBoard (should be in [-10, 10] range)
- Increase exploration: set `entropy_coef = 0.05` in `train_sre_rl.py`
- Train longer (1000+ episodes)

## Next Steps

1. Train base SRE if not done: `python3 main.py --mode sre`
2. Run RL fine-tuning: `python3 run_rl_training.py`
3. Evaluate and compare: `python3 compare_models.py`
4. Check RL_README.md for detailed documentation

## Key Files

- `trainer/train_sre_rl.py` - Main RL trainer (PPO implementation)
- `utils/rl_rewards.py` - Reward function
- `run_rl_training.py` - Launch script
- `compare_models.py` - Evaluation script
- `RL_README.md` - Full documentation
- `revision_plan.md` - Overall revision strategy

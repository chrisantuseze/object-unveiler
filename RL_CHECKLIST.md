# RL Training Execution Checklist

Use this checklist to ensure successful RL training and evaluation.

## ✓ Pre-Training Checklist

### Environment Setup
- [ ] Python 3.8+ installed
- [ ] All dependencies installed: `pip install -r requirements.txt`
- [ ] PyBullet working (test: `python3 -c "import pybullet; print('OK')"`)
- [ ] CUDA available (optional): `python3 -c "import torch; print(torch.cuda.is_available())"`

### Data Preparation
- [ ] Dataset exists: `ls save/pc-ou-dataset/`
- [ ] Pre-trained SRE model exists: `ls save/sre/sre_model_best.pt`
  - If missing: `python3 main.py --mode sre --epochs 100`
- [ ] Config file exists: `ls yaml/bhand.yml`
- [ ] Object assets available: `ls assets/objects/seen/`

### Code Verification
- [ ] Run test suite: `python3 test_rl_setup.py`
- [ ] Check imports: `python3 -c "from trainer.train_sre_rl import train_sre_rl"`
- [ ] Verify environment: `python3 -c "from env.environment import Environment"`

## ✓ Training Execution

### Start Training
- [ ] Launch training: `python3 run_rl_training.py`
- [ ] Verify TensorBoard logs created: `ls runs/`
- [ ] Monitor in real-time: `tensorboard --logdir runs/` (in separate terminal)

### Monitor Progress (check every 50 episodes)
- [ ] Average reward is increasing
- [ ] Episode length is reasonable (not stuck)
- [ ] Success rate improving (should reach >30% by episode 200)
- [ ] No GPU memory errors
- [ ] Checkpoints being saved: `ls save/sre_rl/`

### Expected Training Time
- [ ] ~2-4 hours for 500 episodes (GPU)
- [ ] ~8-12 hours for 500 episodes (CPU)
- [ ] Can interrupt and resume (checkpoints saved every 100 episodes)

## ✓ Post-Training Validation

### Model Checkpoints
- [ ] Final model saved: `ls save/sre_rl/sre_rl_last.pt`
- [ ] Best model saved: `ls save/sre_rl/sre_rl_best.pt`
- [ ] Intermediate checkpoints: `ls save/sre_rl/sre_rl_*.pt`

### Training Logs
- [ ] TensorBoard logs: `ls runs/`
- [ ] Check learning curves in TensorBoard
- [ ] Verify no NaN/Inf values in logs

### Quick Sanity Check
Run this Python snippet:
```python
import torch
from trainer.train_sre_rl import SREActorCritic

args = type('Args', (), {
    'device': torch.device('cpu'),
    'num_patches': 10,
    'patch_size': 64
})()

model = SREActorCritic(args)
model.load_state_dict(torch.load('save/sre_rl/sre_rl_best.pt', map_location='cpu'))
print("✓ Model loads successfully")
```

## ✓ Evaluation Execution

### Run Comparison
- [ ] Execute: `python3 compare_models.py`
- [ ] Generates `comparison_results.png`
- [ ] Generates `comparison_report.txt`
- [ ] Generates `comparison_raw_results.pkl`

### Review Results
- [ ] Open `comparison_results.png` - check all 4 subplots
- [ ] Read `comparison_report.txt` - verify improvement metrics
- [ ] Success rate: RL > SRE-heuristic > Heuristic
- [ ] Strategy divergence > 0 (RL differs from heuristic)

### Expected Results (target metrics)
- [ ] Heuristic success rate: ~40-50%
- [ ] SRE-heuristic success rate: ~50-60%
- [ ] SRE-RL success rate: **≥60%** (goal: 10%+ improvement)
- [ ] Strategy divergence: **≥2 steps per episode**

## ✓ Analysis and Visualization

### Attention Visualization (optional but recommended)
Create attention heatmaps showing what each model focuses on:
```python
# TODO: Add visualization code
# Shows: Heuristic vs SRE-heuristic vs SRE-RL attention patterns
```

### Statistical Significance
- [ ] Run t-test on success rates
- [ ] Compute confidence intervals
- [ ] Document in report

### Case Studies
Identify and document 3-5 examples where:
- [ ] RL outperforms heuristic (show scene + decision)
- [ ] RL makes different choice than heuristic (explain why)
- [ ] Heuristic fails but RL succeeds (demonstrate learning)

## ✓ Paper Integration

### Experimental Section
- [ ] Add RL fine-tuning subsection
- [ ] Include comparison table (3 policies × 4 metrics)
- [ ] Add learning curve figure
- [ ] Add strategy divergence plot

### Results to Report
- [ ] Success rate improvement: X% over heuristic
- [ ] Sample efficiency: Y episodes to convergence
- [ ] Strategy divergence: Z decisions per episode differ
- [ ] Qualitative improvements (accessibility, stability)

### Figures to Create
- [ ] Figure X: Training curves (reward, success rate)
- [ ] Figure Y: Comparison bar charts (from compare_models.py)
- [ ] Figure Z: Attention visualization (heuristic vs RL)
- [ ] Table X: Quantitative comparison

### Address Reviewer Comments
- [ ] **Reviewer 3**: "How model accounts for accessibility/stability"
  → Show reward components explicitly model these
  → Demonstrate learned trade-offs
  
- [ ] **Reviewer 5**: "Heavy reliance on heuristic supervision"
  → Show RL diverges from heuristic
  → Demonstrate performance improvement
  → Prove learning beyond imitation

- [ ] **Associate Editor**: "Zero-shot transfer claim"
  → Explain how RL learns reachability-aware policies
  → Show better generalization potential

## ✓ Troubleshooting Checklist

If training fails:
- [ ] Check TensorBoard - are rewards all zeros?
- [ ] Verify environment resets properly
- [ ] Check reward scaling (should be -10 to +10)
- [ ] Increase exploration (entropy_coef = 0.05)
- [ ] Reduce learning rate (lr = 0.00001)

If no learning occurs:
- [ ] Train longer (1000 episodes)
- [ ] Verify reward function is working (print components)
- [ ] Check action distribution (not collapsing?)
- [ ] Try simpler scenarios first (fewer objects)

If evaluation fails:
- [ ] Verify all model files exist
- [ ] Check model device (CPU vs GPU mismatch)
- [ ] Reduce n_episodes in compare_models.py
- [ ] Run in debug mode with smaller batch

## ✓ Final Deliverables

### For Resubmission
- [ ] Trained RL model: `save/sre_rl/sre_rl_best.pt`
- [ ] Comparison plots: `comparison_results.png`
- [ ] Comparison report: `comparison_report.txt`
- [ ] Raw data: `comparison_raw_results.pkl`
- [ ] Updated paper with RL results
- [ ] Response to reviewers citing new experiments

### For Reproducibility
- [ ] Training script: `run_rl_training.py`
- [ ] Evaluation script: `compare_models.py`
- [ ] Documentation: `RL_README.md`
- [ ] Architecture diagram: `RL_ARCHITECTURE.md`
- [ ] This checklist: `RL_CHECKLIST.md`

## ✓ Timeline

### Week 1
- [ ] Day 1: Setup and test
- [ ] Day 2-3: Train RL model
- [ ] Day 4: Run evaluation
- [ ] Day 5: Analyze results

### Week 2
- [ ] Day 1-2: Create visualizations
- [ ] Day 3-4: Write paper sections
- [ ] Day 5: Review and revise

---

## Quick Command Reference

```bash
# Test setup
python3 test_rl_setup.py

# Train RL
python3 run_rl_training.py

# Monitor training
tensorboard --logdir runs/

# Evaluate
python3 compare_models.py

# Check results
cat comparison_report.txt
open comparison_results.png
```

---

**Last Updated**: February 10, 2026  
**Status**: Ready for execution  
**Estimated Time to Complete**: 3-5 days

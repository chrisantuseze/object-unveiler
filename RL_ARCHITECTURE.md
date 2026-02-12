# RL Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         RL Fine-tuning Pipeline                              │
└─────────────────────────────────────────────────────────────────────────────┘

                                 TRAINING PHASE
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                               │
│  ┌─────────────┐                                                             │
│  │ Environment │ ← Reset with random scene                                   │
│  │  (PyBullet) │                                                             │
│  └──────┬──────┘                                                             │
│         │                                                                     │
│         │ Observation (RGB, depth, masks)                                    │
│         ↓                                                                     │
│  ┌─────────────────┐                                                         │
│  │  Segmentation   │ ← MaskRCNN                                              │
│  │  (object masks) │                                                         │
│  └────────┬────────┘                                                         │
│           │                                                                   │
│           │ Object masks, bboxes, target                                     │
│           ↓                                                                   │
│  ┌──────────────────────────────────────────────────────┐                   │
│  │           SREActorCritic Network                      │                   │
│  │                                                        │                   │
│  │  ┌─────────────────────────────────────────────┐     │                   │
│  │  │  Actor (Spatial Relationship Encoder)       │     │                   │
│  │  │  ┌──────────────────────────────────────┐   │     │                   │
│  │  │  │ ResNet18 Feature Extractor           │   │     │                   │
│  │  │  ├──────────────────────────────────────┤   │     │                   │
│  │  │  │ Transformer Layers (4 layers)        │   │     │                   │
│  │  │  ├──────────────────────────────────────┤   │     │                   │
│  │  │  │ Output Projection → [B, N] logits    │   │     │                   │
│  │  │  └──────────────────────────────────────┘   │     │                   │
│  │  │  Pre-trained on heuristic data              │     │                   │
│  │  └─────────────────────────────────────────────┘     │                   │
│  │                          │                            │                   │
│  │                          │ Action logits              │                   │
│  │                          ↓                            │                   │
│  │  ┌─────────────────────────────────────────────┐     │                   │
│  │  │  Critic (Value Network)                     │     │                   │
│  │  │  ┌──────────────────────────────────────┐   │     │                   │
│  │  │  │ Linear(N → 256)                      │   │     │                   │
│  │  │  ├──────────────────────────────────────┤   │     │                   │
│  │  │  │ Linear(256 → 128)                    │   │     │                   │
│  │  │  ├──────────────────────────────────────┤   │     │                   │
│  │  │  │ Linear(128 → 1) → State value        │   │     │                   │
│  │  │  └──────────────────────────────────────┘   │     │                   │
│  │  │  Trained from scratch                       │     │                   │
│  │  └─────────────────────────────────────────────┘     │                   │
│  └──────────────────────────────────────────────────────┘                   │
│           │                            │                                     │
│           │ Action (object idx)        │ Value estimate                      │
│           ↓                            ↓                                     │
│  ┌────────────────────┐       ┌────────────────┐                            │
│  │ Sample from        │       │ Store in       │                            │
│  │ action distribution│       │ PPO Memory     │                            │
│  └─────────┬──────────┘       └────────────────┘                            │
│            │                                                                  │
│            │ Execute grasp action                                            │
│            ↓                                                                  │
│  ┌─────────────────┐                                                         │
│  │  Environment    │                                                         │
│  │  Step           │                                                         │
│  └────────┬────────┘                                                         │
│           │                                                                   │
│           │ Next observation, grasp info                                     │
│           ↓                                                                   │
│  ┌─────────────────────────────────────────────────────────────┐            │
│  │              Reward Computation                              │            │
│  │  ┌────────────────────────────────────────────────────────┐ │            │
│  │  │ • Distance to target reward        (+2.0 weight)       │ │            │
│  │  │ • Path clearance reward            (+1.5 weight)       │ │            │
│  │  │ • Grasp accessibility reward       (+1.0 weight)       │ │            │
│  │  │ • Stability penalty                (-3.0 weight)       │ │            │
│  │  │ • Grasp success/failure            (+1.0/-2.0)         │ │            │
│  │  │ • Collision penalty                (-1.0)              │ │            │
│  │  │ • Efficiency bonus                 (+0.5)              │ │            │
│  │  │ • Target reached bonus             (+10.0)             │ │            │
│  │  └────────────────────────────────────────────────────────┘ │            │
│  │                           ↓                                  │            │
│  │                  Total Reward = Σ components                │            │
│  └─────────────────────────────────────────────────────────────┘            │
│           │                                                                   │
│           │ Reward                                                           │
│           ↓                                                                   │
│  ┌────────────────┐                                                          │
│  │ Store in       │                                                          │
│  │ PPO Memory     │                                                          │
│  └────────┬───────┘                                                          │
│           │                                                                   │
│           │ After N episodes...                                              │
│           ↓                                                                   │
│  ┌────────────────────────────────────────────────────┐                     │
│  │         PPO Update (K epochs)                       │                     │
│  │  ┌──────────────────────────────────────────────┐  │                     │
│  │  │ 1. Compute advantages: A = R - V             │  │                     │
│  │  │ 2. Compute policy ratio: r = π_new/π_old    │  │                     │
│  │  │ 3. Clipped surrogate loss:                   │  │                     │
│  │  │    L = min(r·A, clip(r, 1±ε)·A)            │  │                     │
│  │  │ 4. Value loss: MSE(V, R)                     │  │                     │
│  │  │ 5. Entropy bonus: -H(π)                      │  │                     │
│  │  │ 6. Total loss: L_actor + 0.5·L_critic +     │  │                     │
│  │  │                 0.01·L_entropy                │  │                     │
│  │  │ 7. Backprop & update weights                 │  │                     │
│  │  └──────────────────────────────────────────────┘  │                     │
│  └────────────────────────────────────────────────────┘                     │
│           │                                                                   │
│           └───────────────────┐                                              │
│                               │ Repeat for 500-1000 episodes                 │
│                               └──────────────────────────────────────────────┘

                                EVALUATION PHASE
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                               │
│         Test Scene                                                           │
│              ↓                                                               │
│   ┌──────────┴──────────┬──────────────┬──────────────┐                    │
│   │                     │              │              │                     │
│   ↓                     ↓              ↓              ↓                     │
│ ┌────────┐        ┌──────────┐   ┌─────────┐   ┌──────────┐               │
│ │Heuristic│       │SRE        │   │SRE + RL │   │Ground    │               │
│ │Baseline │       │(trained)  │   │(finetuned)│ │Truth     │               │
│ └────┬───┘        └─────┬────┘   └────┬────┘   └────┬─────┘               │
│      │                  │              │             │                      │
│      │   Select object to remove       │             │                      │
│      ↓                  ↓              ↓             ↓                      │
│   Action            Action         Action       Optimal                     │
│      │                  │              │             │                      │
│      └──────────────────┴──────────────┴─────────────┘                     │
│                          │                                                   │
│                          ↓                                                   │
│                 ┌────────────────┐                                           │
│                 │ Compare:       │                                           │
│                 │ • Success rate │                                           │
│                 │ • Steps taken  │                                           │
│                 │ • Collisions   │                                           │
│                 │ • Divergence   │                                           │
│                 └────────┬───────┘                                           │
│                          │                                                   │
│                          ↓                                                   │
│                 ┌────────────────┐                                           │
│                 │ Generate:      │                                           │
│                 │ • Plots        │                                           │
│                 │ • Report       │                                           │
│                 │ • Statistics   │                                           │
│                 └────────────────┘                                           │
└─────────────────────────────────────────────────────────────────────────────┘

                            KEY INSIGHT
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                               │
│  Heuristic Policy:  Distance(object, target) → argmin                        │
│      ↓                                                                       │
│  SRE (trained):     Learns to imitate heuristic from data                    │
│      ↓                                                                       │
│  SRE + RL:          Discovers better strategies by:                          │
│                     - Balancing accessibility, stability, distance           │
│                     - Learning from trial-and-error                          │
│                     - Optimizing for episode success, not just distance      │
│                                                                               │
│  Result: RL policy diverges from heuristic when it finds better actions!     │
│                                                                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Files and Their Roles

```
trainer/train_sre_rl.py
├── SREActorCritic         # Main network architecture
├── PPOMemory              # Trajectory storage
├── RLEnvironmentWrapper   # Gym-style interface
├── train_sre_rl()         # Training loop
└── update_policy_ppo()    # PPO optimization

utils/rl_rewards.py
├── compute_episode_reward()      # Main reward function
├── compute_target_distance()     # Distance-based reward
├── compute_stability_penalty()   # Stability checking
├── compute_path_clearance()      # Blocking objects
└── compute_accessibility()       # Grasp ease

compare_models.py
├── load_models()          # Load all 3 policies
├── run_episode()          # Execute policy
├── compute_statistics()   # Aggregate metrics
└── plot_comparison()      # Generate visualizations

main.py (modified)
└── mode='sre-rl'          # New training mode
```

## Training Workflow

```
Step 1: Pre-train SRE on heuristic data
   python3 main.py --mode sre --epochs 100
                    ↓
Step 2: Fine-tune with RL
   python3 run_rl_training.py
                    ↓
Step 3: Evaluate and compare
   python3 compare_models.py
                    ↓
Step 4: Analyze results
   - Check TensorBoard for learning curves
   - Review comparison_report.txt
   - Examine attention visualizations
```

## Expected Learning Curve

```
Average Reward over Episodes

20 ┤                                                      ╭─────
   │                                                 ╭────╯
15 ┤                                            ╭────╯
   │                                       ╭────╯
10 ┤                                  ╭────╯
   │                             ╭────╯
 5 ┤                        ╭────╯
   │                   ╭────╯
 0 ┼──────────────╭────╯
   │         ╭────╯
-5 ┤    ╭────╯
   │╭───╯
-10┤╯
   └┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬
    0   50  100  150  200  250  300  350  400  450  500
                         Episodes

   Phase 1 (0-100):   Exploration, high variance
   Phase 2 (100-300): Learning, reward increases
   Phase 3 (300+):    Convergence, stable policy
```

# Unveiler Revision Plan - ICRA 2026 Resubmission

**Target:** Address core criticisms around novelty and learning contribution

**Focus Areas:** 
1. Real learning beyond heuristic supervision
3. Theoretical contribution on decomposition benefits

---

## 1. Add Real Learning Beyond the Heuristic

### Goal
Demonstrate that the model learns strategies beyond the heuristic expert and can improve upon it.

### Implementation Steps

#### 1.1 Fine-tune SRE with Reinforcement Learning
**Timeline: 2-3 weeks**

- [ ] **Design RL formulation**
  - State: Object masks + heightmap + target encoding (current SRE inputs)
  - Action: Next object to remove (discrete action space over N objects)
  - Reward: Dense reward based on:
    - Distance reduction to target per step
    - Stability penalty (objects knocked over)
    - Episode success (target retrieved)
    - Step efficiency bonus
  
- [ ] **Implementation approach**
  - Initialize policy network with pre-trained SRE weights
  - Use PPO or SAC for fine-tuning
  - Train in simulation with diverse clutter scenarios
  - Keep action decoder frozen (or fine-tune separately)

- [ ] **Key files to modify**
  ```
  policy/sre_model.py          # Add RL training mode
  trainer/sre_trainer.py       # Add RL fine-tuning loop
  main.py                      # Add RL training script
  ```

- [ ] **Experiments to run**
  - Baseline: Heuristic-only policy (Algorithm 1)
  - Comparison 1: SRE trained on heuristic (current)
  - Comparison 2: SRE trained on heuristic + RL fine-tuned
  - Metrics: Success rate, steps to target, stability violations

#### 1.2 Identify Cases Where Learning Outperforms Heuristic
**Timeline: 1 week**

- [ ] **Create challenging scenarios**
  - Dense clutter (10+ objects)
  - Multiple viable removal sequences
  - Cases where shortest-path heuristic fails (narrow passages, stability)
  - Partial observability (occluded target)

- [ ] **Quantitative analysis**
  - Success rate gap: RL-finetuned vs heuristic
  - Strategy divergence: Count steps where RL chooses different object than heuristic
  - Efficiency gain: Average steps to success

- [ ] **Qualitative analysis**
  - Visualize attention weights: What spatial relationships does RL-model prioritize?
  - Case studies: 3-5 scenarios with side-by-side comparison
  - Show emergent behaviors (e.g., clearing stable base first, exploiting gaps)

#### 1.3 Self-Supervised Learning Alternative (if RL challenging)
**Timeline: 1-2 weeks**

- [ ] **Contrastive learning for spatial reasoning**
  - Augment scenes with random object permutations
  - Train SRE to distinguish valid vs invalid removal sequences
  - Use hindsight relabeling: failed trajectories as negative examples

- [ ] **Data augmentation curriculum**
  - Generate harder scenarios than heuristic training set
  - Use difficulty metrics: clutter density, target depth, stability
  - Show model generalizes to out-of-distribution scenarios

---

## 3. Theoretical Contribution

### Goal
Provide formal analysis of why decomposition improves learning in this task class.

### Implementation Steps

#### 3.1 Sample Complexity Analysis
**Timeline: 2-3 weeks**

- [ ] **Formalize the problem**
  - State space: S = {object configurations}
  - Action space: A = {grasp poses} 
  - Hierarchical decomposition: π(a|s) = π_action(a|s, o*) · π_spatial(o*|s)
  - End-to-end baseline: π_mono(a|s)

- [ ] **Theoretical claims to prove/support**
  - **Claim 1**: Decomposition reduces effective action space
    - Spatial module reduces from |A| to |O| decisions (O = # objects)
    - Action module operates in reduced space conditioned on o*
    - Combined complexity: O(|O| + |A_o|) vs O(|A|) where |A| >> |O| · |A_o|
  
  - **Claim 2**: Sample efficiency gain
    - Spatial reasoning reuses across different grasp executions
    - Action decoder reuses across different target objects
    - Use PAC learning bounds or VC dimension arguments

  - **Claim 3**: Generalization bound
    - Show decomposition provides inductive bias
    - Spatial module generalizes across object counts/types
    - Action module generalizes across object geometries

- [ ] **Empirical validation of theory**
  - Plot learning curves: samples needed to reach 80% success
  - Vary object count, measure sample efficiency gap
  - Ablation: train with limited data, compare decomposed vs end-to-end

#### 3.2 Approximation Error Analysis
**Timeline: 1 week**

- [ ] **Decomposition error bounds**
  - Analyze: π*(a|s) vs π_action(a|s,o*) · π_spatial(o*|s)
  - When does factorization lose optimality?
  - Show error is bounded by spatial reasoning accuracy

- [ ] **Write-up structure**
  ```
  Section: Theoretical Analysis
  - 3.1 Problem Formalization
  - 3.2 Sample Complexity of Decomposition
  - 3.3 Approximation Guarantees
  - 3.4 Empirical Validation
  ```

#### 3.3 Comparison to Prior Hierarchical Methods
**Timeline: 1 week**

- [ ] **Position against related work**
  - Options framework (Sutton et al.)
  - Goal-conditioned RL
  - Transporter Networks (spatial action maps)
  - Slot Attention (object-centric learning)

- [ ] **What's formally different**
  - Explicit spatial reasoning module with transformer attention
  - Decoupling object selection from grasp execution
  - Rotation equivariance in action space

- [ ] **Add comparison table**
  | Method | Spatial Reasoning | Action Decomposition | Sample Efficiency |
  |--------|-------------------|---------------------|-------------------|
  | End-to-end | Implicit | None | Baseline |
  | Transporter | Per-pixel | Pick-place coupled | Better |
  | Unveiler (ours) | Object-centric | Decoupled | Best |

---

## Additional Improvements (Quick Wins)

### Fix Video Issues
**Timeline: 1 day**
- [ ] Align RGB/perspective in visualization
- [ ] Correct color channel ordering (BGR→RGB)
- [ ] Add overlays showing SRE attention weights
- [ ] Include side-by-side heuristic vs learned comparisons

### Expand Real-World Validation (if time permits)
**Timeline: 2-3 weeks**
- [ ] Increase trials to 50+ (currently small number)
- [ ] Test on varied objects (shape/size/weight)
- [ ] Quantify sim-to-real gap explicitly
- [ ] Add reachability-aware metrics

### Strengthen Writing
**Timeline: 3-4 days**
- [ ] Add "What is New" paragraph in intro
- [ ] Explicit contributions list
- [ ] Failure analysis section
- [ ] Ablation table comparing all variants

---

## Timeline Summary

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| RL fine-tuning implementation | 2-3 weeks | Trained RL-enhanced SRE |
| Challenging scenarios & analysis | 1 week | Comparison results |
| Theoretical formalization | 2-3 weeks | Theory section draft |
| Empirical validation of theory | 1 week | Learning curve plots |
| Paper revision & writing | 1 week | Revised manuscript |
| **Total** | **7-9 weeks** | Resubmission-ready paper |

---

## Success Criteria

### For RL Learning Component (Suggestion 1)
- [ ] RL-finetuned model achieves ≥10% higher success rate than heuristic in challenging scenarios
- [ ] Identify ≥3 clear strategy differences with qualitative analysis
- [ ] Attention visualizations show learned spatial priorities

### For Theoretical Contribution (Suggestion 3)
- [ ] Formal sample complexity bound with proof sketch
- [ ] Empirical validation showing 2-5x sample efficiency gain
- [ ] Clear positioning against prior hierarchical methods

### For Acceptance
- [ ] Addresses all 3 major reviewer concerns directly
- [ ] Adds substantial technical depth beyond original submission
- [ ] Provides generalizable insights for manipulation community

---

## Next Steps

1. **Start with RL implementation** (highest impact, longest timeline)
   - Begin with `policy/sre_model.py` modifications
   - Set up RL training environment

2. **Parallel: Draft theory section** (can work independently)
   - Formalize problem and decomposition
   - Work out sample complexity arguments

3. **After RL converges: Run comparison experiments**
   - Generate challenging scenarios
   - Collect performance metrics

4. **Final: Integrate everything into paper revision**
   - New theory section
   - RL results in experiments
   - Updated related work

---

## Files to Create/Modify

### New Files
- `trainer/rl_trainer.py` - RL fine-tuning loop
- `utils/rl_rewards.py` - Reward function definitions
- `experiments/challenging_scenarios.py` - Test case generator
- `theory/sample_complexity.py` - Theoretical analysis utils

### Modified Files
- `policy/sre_model.py` - Add RL training mode
- `main.py` - Add RL training entry point
- `eval_agent.py` - Add RL-enhanced evaluation
- Paper manuscript - Major revision

---

## Questions to Resolve

1. **RL algorithm choice**: PPO (stable, sample efficient) vs SAC (off-policy, better exploration)?
2. **Reward shaping**: Dense vs sparse? Need to balance with heuristic guidance.
3. **Theory depth**: Full proofs vs proof sketches? (Depends on target venue - ICRA vs journal)
4. **Computational resources**: GPU time for RL training? Estimated 3-5 days continuous training.

---

*Created: February 10, 2026*
*Target resubmission: IROS 2026 (March deadline) or T-RO journal*

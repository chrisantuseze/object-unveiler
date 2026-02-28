# IROS Revision Recommendations
> Based on ICRA 2026 reviewer feedback. February 28, 2026.

---

## 0. Most Urgent: Fill in Table II

Table II is entirely `*` for Unveiler. This is the single most dangerous issue.
The IL results (no RL) should already be finalized — fill those in immediately.
Only the IL+RL column may remain `*` while RL training is still running.

---

## 1. Narrow the Zero-Shot Transfer Claim

**Affects: Abstract + Section VI-G**

The current abstract overreaches by implying the full system transfers without
any real-world adaptation. The Action Decoder requires workspace bound
calibration, which is real-world adaptation even if not fine-tuning.

### Abstract — replace:
> *"Additionally, we demonstrate successful zero-shot transfer of our
> simulation-trained policy to a real robotic system without any real-world
> fine-tuning."*

### With:
> *"Additionally, we demonstrate that the SRE's spatial reasoning transfers
> zero-shot to real scenes, and validate the full system on a physical robot
> requiring only geometric workspace calibration — no learned components are
> retrained."*

### Section VI-G — add after the opening paragraph:
> *"Critically, only the workspace bounds are adapted to the physical setup —
> a one-time geometric calibration. No learned component (neither the SRE nor
> the Action Decoder) is retrained or fine-tuned on real data. This confirms
> that the kinematics of the deployment arm, being encoded only in the workspace
> bounds and not in the learned policy, do not introduce a domain gap for the
> SRE."*

**Why this works:** It turns the reviewer's criticism into a design feature.
The separation means the arm-specific constraint is cleanly isolated to a
non-learned parameter.

---

## 2. Add Section VI-H: SRE Scene Reasoning Evaluation on Real Images

**This is the most important new addition.**
Place it between Section VI-G (Real Robot) and the Conclusion.

### Experimental setup:
- Collect 50–100 real table scenes, photographed top-down.
- For each scene: annotate ground-truth first-removal object (human expert
  annotation, or run the simulation heuristic on the real heightmap as oracle).
- Evaluate three models:
  - **SRE** (yours)
  - **GPT-4o** — prompt with scene image + target crop, ask which object to
    remove first
  - **CLIP** — rank objects by cosine similarity to the target crop
- Report top-1 object selection accuracy in a small table, broken down by
  scene density (2–4 objects, 4–6 objects).

### How to frame it (critical):
Do NOT frame this as a fallback for failed robot experiments. Frame it as:

> *"To isolate the SRE's spatial reasoning from execution noise and evaluate
> its transferability independently, we evaluate object-selection accuracy
> directly on real scenes. This decoupled evaluation is natural given the
> factored architecture: the SRE's output — which object to remove — can be
> assessed independently of any physical arm."*

This makes the modularity of your architecture the justification for the
evaluation, not a limitation.

---

## 3. RL Finetuning: Sharpen the Claims Once Numbers Land

**Affects: Contribution bullet 4 + Section VI-E**

### Contribution bullet — once results are ready, add specifics:
Replace the current vague claim with:
> *"RL-enhanced spatial reasoning using PPO to fine-tune the
> imitation-pretrained SRE, improving over the IL baseline by X% in dense,
> fully occluded scenes (9–12 objects) and recovering cases where the heuristic
> fails."*

The 9–12 object / full occlusion row is the most compelling comparison point —
that is where the heuristic collapses to 20% and where RL should show the
largest delta.

### Section VI-E (Spatial Reasoning and Interpretability) — add:
2–3 qualitative examples showing specific episodes where RL chose a different
removal sequence than the heuristic and succeeded. This directly addresses
reviewer 5's demand for insight into *why* the approach works — show the
emergent strategy visually.

---

## 4. Fix the Dangling Claim in Section VI-G

Section VI-G currently ends with:
> *"This quantitatively validates the robustness of the learned Action Decoder
> to variations and noise in the heightmap source."*

With `*%` and `*` trials in the text, this sentence does no work and
actively undermines credibility. Until those numbers are filled in:
- Either cut this sentence entirely, or
- Replace with a qualitative description of what was observed.

---

## 5. What NOT to Change

- **Section V (theory)** is strong as-is. It already addresses reviewer 5's
  "no formal analysis" comment directly. Leave it.
- **Ablation tables (III and IV)** are solid. Leave them.
- **Table I (computational efficiency)** is a good differentiator. Keep it
  prominent.
- **Related work** adequately covers the space. Do not expand it — stay within
  page limits.

---

## 6. Supplemental Video

### Recommendation: Keep both clips. Re-edit with explicit title cards.

Do **not** remove the Barrett hand simulation clips. A reviewer who notices
they were cut (or recalls seeing them in a prior submission) will assume you
are hiding the sim-real gap. That is worse than the gap itself.

The right move is to reframe the contrast between the two robots as a
*demonstration of your modularity claim*, not an inconsistency:

- The sim clips show the Barrett hand grasping from all directions
  → evidence that the **SRE is robot-agnostic** (trained on a floating hand,
  reasoning is about scene geometry, not arm kinematics)
- The Dofbot clips show a constrained serial arm executing the same policy
  → evidence that **only the non-learned workspace bounds need recalibration**

A reviewer seeing both robots with explicit framing will read it as proof of
your claim, not a contradiction.

---

### Exact title card texts to use in the video

**Before the real robot (Dofbot-Pro) clips:**

> **Real-World Deployment — Dofbot-Pro**
> Same policy weights as simulation. No retraining.
> Only workspace bounds recalibrated.

---

**Between the real-robot and simulation clips (transition card):**

> **Training was done on a different robot entirely.**
> The SRE outputs an object index — not arm commands.
> It has no knowledge of kinematics.

---

**Before the simulation (Barrett hand) clips:**

> **Training Environment — Simulation (Barrett Hand)**
> Full 360° workspace access. Fundamentally different kinematics.
> The same SRE reasoning transfers zero-shot.

---

### Why this framing works

The reviewer's concern is: *"you trained on a robot with no kinematic
constraints; how does that transfer?"* These cards answer it directly and
preemptively: the SRE has no kinematic inputs, so there are no arm-specific
constraints to transfer. The only thing that changes between the Barrett hand
and the Dofbot is the workspace bounds — a non-learned parameter. Showing
both robots side-by-side with this explanation is stronger evidence than
showing the Dofbot alone, because it visually demonstrates the separation
between reasoning and execution.

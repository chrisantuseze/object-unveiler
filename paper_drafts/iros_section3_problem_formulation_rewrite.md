# IROS Drop-in Rewrite: Section 3 (Problem Formulation and Decomposition)

> Goal: preserve your current content but compress it and add a single “decomposition equation” bridge so Section 4.4 can be shorter.

## 3 Problem Formulation and Decomposition

We study **sequential target retrieval from dense clutter** using a top-down RGB-D perception stack and a push–grasp primitive. Given an RGB scene image $I$ and corresponding depth heightmap $I_g$ (constructed from the fused RGB-D point cloud), instance segmentation (Mask R-CNN) produces a set of object masks $\mathcal{M}=\{M_1,\dots,M_{N}\}$ and a target index $t$. We use heightmaps rather than raw depth images because they provide a robot-centric, top-down representation that reduces perspective distortion and yields a consistent action reference frame.

**Segmentation mismatch.** In clutter, the number of detected instances may differ from the true object count. We denote by $N$ the number of predicted masks at the current step; Unveiler predicts and executes **one action per step**, so each removal typically improves visibility and reduces segmentation errors over time.

**Action primitive and constraints.** Following Kiatos et al. [8], the robot executes a push–grasp primitive under the constraints: (i) fixed end-effector height during execution, (ii) constant aperture during the push phase, (iii) heightmap boundaries aligned with workspace limits, and (iv) a $0.5\,\text{m}\times0.5\,\text{m}$ tabletop workspace.

### Decomposed sequential decision problem

At each decision step, the robot must (a) choose which visible object to remove next and (b) execute a continuous push–grasp action to remove it. Unveiler factorizes this into a discrete **object-selection** policy (Spatial Relationship Encoder; SRE) and a conditional **action-execution** policy (Action Decoder):

\[
\pi(a\mid s) \;=\; \sum_{o\in\{1,\dots,N\}} \pi_{\text{SRE}}(o\mid s)\;\pi_{\text{AD}}(a\mid s,o),
\]

where state $s$ includes the scene observation and object masks $\mathcal{M}$ with target index $t$, $o$ indexes an object instance, and $a\in\mathbb{R}^4$ parameterizes the push–grasp primitive. Since each step removes one object, the planning horizon satisfies $H\le N$. This decomposition enables the SRE to focus on **object-centric spatial reasoning** (which obstacle is most critical), while the Action Decoder focuses on robust, rotation-invariant execution for the selected object.

> Theoretical implication (preview): this factorization reduces the learning problem to multiclass object selection and yields an interpretable bound relating end-task performance to SRE selection error (Section 4.4).

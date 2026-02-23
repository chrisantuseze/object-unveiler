# Supplementary: Derivations for Section 4.4

---

## S.1 Notation and Setup

We model sequential target retrieval as a finite-horizon MDP with horizon $H \leq N$ (one object removed per step). At each step $h \in \{1, \dots, H\}$:

- State $s_h$: depth heightmap + instance masks $\mathcal{M} = \{M_1, \dots, M_N\}$ + target index $t$.
- High-level action: object index $o \in \{1, \dots, N\}$ selected by $\pi_{\text{SRE}}(o \mid s)$.
- Low-level action: push–grasp parameters $a \in \mathbb{R}^4$ produced by $\pi_{\text{AD}}(a \mid s, o)$.
- Reward $r(s, a) \in [0, 1]$, bounded.
- Factored policy: $\pi(a \mid s) = \sum_o \pi_{\text{SRE}}(o \mid s)\, \pi_{\text{AD}}(a \mid s, o)$.

The Action Decoder is trained independently and held fixed; all learning analysis below concerns $\pi_{\text{SRE}}$ only.

---

## S.2 Derivation of the Sample Complexity Comparison

**The SRE case.** With $\pi_{\text{AD}}$ fixed, the optimal high-level action at each state is
$$o^*(s) = \arg\max_{o} \;\mathbb{E}_{a \sim \pi_{\text{AD}}(\cdot \mid s,o)}\bigl[r(s,a) + V^*(s')\bigr].$$
Training labels are provided by the heuristic (Algorithm 1), which approximates $o^*(s)$. Learning $\pi_{\text{SRE}}$ therefore reduces to **$N$-way multiclass classification** on object-centric feature vectors.

For multiclass classification into $N$ classes, the standard agnostic PAC bound gives a sample requirement
$$n = \mathcal{O}\!\left(\frac{\log N + \log(1/\delta)}{\varepsilon^2}\right).$$
The $\log N$ term appears because the output space has $N$ classes; the confidence term $\log(1/\delta)$ is typically negligible. Dropping constants and the confidence term gives the simplified form in the main paper:
$$n_{\text{SRE}} = \mathcal{O}\!\left(\frac{\log N}{\varepsilon^2}\right).$$

**The monolithic baseline case.** An end-to-end policy maps states directly to pixel-space action maps of size $W \times W$ at $K$ discrete orientations, i.e., a $W^2 K$-dimensional output. For regression or dense prediction over this output space, sample complexity scales linearly with the output dimension (from standard results for linear function classes or convolutional networks with bounded weight norms):
$$n_{\text{mono}} = \mathcal{O}\!\left(\frac{W^2 K}{\varepsilon^2}\right).$$

**Numerical comparison.** With $N = 12$, $W = 224$, $K = 16$:
$$\frac{n_{\text{mono}}}{n_{\text{SRE}}} \approx \frac{W^2 K}{\log N} = \frac{224^2 \times 16}{\log 12} \approx \frac{802{,}816}{3.58} \approx 224{,}000.$$
This order-of-magnitude difference in the numerator is the formal basis for the data efficiency gap between Unveiler (40K demonstrations) and end-to-end RL methods (10M+ samples).

---

## S.3 Derivation of the Error Propagation Bound

**Setup.** Let $\pi^*$ denote the policy that always selects the optimal object $o^*(s_h)$ at each step (using the same $\pi_{\text{AD}}$). Define the per-step value gap:
$$\delta_h(s) = \mathbb{E}_{a \sim \pi_{\text{AD}}(\cdot \mid s, o^*(s))}\bigl[r + V^*(s')\bigr] - \mathbb{E}_{a \sim \pi_{\text{AD}}(\cdot \mid s, \hat{o}(s))}\bigl[r + V(s')\bigr],$$
where $\hat{o}(s) \sim \pi_{\text{SRE}}(\cdot \mid s)$ is the SRE's selection and $V$ is the value under $\pi$.

**Assumptions.**
1. The SRE selects a suboptimal object with probability at most $\epsilon_{\text{SRE}}$ per step: $\Pr[\hat{o}(s_h) \neq o^*(s_h)] \leq \epsilon_{\text{SRE}}$.
2. A wrong object pick costs at most $\Delta$ in expected one-step value: $\mathbb{E}[\delta_h(s) \mid \hat{o} \neq o^*] \leq \Delta$.
3. $\Delta \leq 1$ (bounded by reward range).

**Derivation.** Apply a step-wise performance decomposition:
$$J(\pi^*) - J(\pi) = \sum_{h=1}^{H} \mathbb{E}_{s_h}\bigl[\delta_h(s_h)\bigr].$$
At each step, $\delta_h(s) = 0$ when $\hat{o}(s) = o^*(s)$ (correct selection), and $\delta_h(s) \leq \Delta$ otherwise. Therefore:
$$\mathbb{E}_{s_h}[\delta_h(s_h)] \leq \Delta \cdot \Pr[\hat{o}(s_h) \neq o^*(s_h)] \leq \Delta \cdot \epsilon_{\text{SRE}}.$$
Summing over $H$ steps:
$$J(\pi^*) - J(\pi) \leq \sum_{h=1}^H \Delta \cdot \epsilon_{\text{SRE}} = H \cdot \Delta \cdot \epsilon_{\text{SRE}}.$$

**Adding the execution error term.** The above assumes $\pi_{\text{AD}}$ is the same for both $\pi^*$ and $\pi$, so it cancels. When $\pi_{\text{AD}}$ is itself suboptimal relative to a perfect executor, an additional term $\epsilon_{\text{exec}}$ appears additively:
$$J(\pi^*) - J(\pi) \leq H \cdot \Delta \cdot \epsilon_{\text{SRE}} + \epsilon_{\text{exec}}.$$
This term is bounded by the performance gap between the **Heur** condition (oracle object selection, same $\pi_{\text{AD}}$) and perfect task performance, which is directly readable from Table 2.

**Key implication.** The bound is linear in $\epsilon_{\text{SRE}}$, not polynomial or exponential. This confirms that selection and execution errors do not compound — improving either module independently improves overall performance monotonically.

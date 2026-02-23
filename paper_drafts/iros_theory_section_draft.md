# IROS Theory Section Draft (Drop-in)

> Intended size: ~0.5–0.6 column pages. Two simple displayed equations.
>
> Placement: after **Section 4.3 Data Collection** as **Section 4.4**.

## 4.4 Properties of the Decomposed Policy

The factorization $\pi = \pi_{\text{SRE}} \cdot \pi_{\text{AD}}$ defined in Section 3 is not merely an architectural convenience. It induces three properties with direct implications for how the system learns and how failures can be diagnosed.

*Reduced learning complexity.* With the Action Decoder held fixed, training the SRE reduces to an $N$-way classification over object crops. The number of demonstrations required to achieve error $\leq \varepsilon$ scales with the log of the decision space, whereas a monolithic policy mapping pixels to grasp parameters scales with the size of the action map:
$$n_{\text{SRE}} \;=\; \mathcal{O}\!\left(\frac{\log N}{\varepsilon^2}\right) \quad \ll \quad n_{\text{mono}} \;=\; \mathcal{O}\!\left(\frac{W^2 K}{\varepsilon^2}\right),$$
where $N \leq 12$ objects, $W = 224$ pixels, and $K = 16$ orientations. Numerically, $\log 12 \approx 3.6$ versus $224^2 \times 16 \approx 800{,}000$ — a difference of $\sim\!220{,}000\times$ in the numerator, consistent with the gap between Unveiler's 40K demonstrations and the 10M+ typically required by end-to-end methods.

*Object-centric inductive bias.* The SRE architecture encodes structure that a flat policy must discover implicitly. The target is always the cross-attention query, anchoring learned representations to target-relative spatial context from the first update. Per-object crops make the model equivariant to instance ordering, so the same selection behavior generalizes across object counts and arrangements not seen during training. Both priors are direct consequences of the factored design, not post-hoc additions.

*Additive error decomposition.* Because selection and execution are separate, their failure modes are also separate. If the SRE misselects on a fraction $\epsilon_{\text{SRE}}$ of steps and each mistake costs at most $\Delta$ in expected progress, the performance gap to an ideal policy is bounded by
$$J(\pi^*) - J(\pi) \;\leq\; H \cdot \Delta \cdot \epsilon_{\text{SRE}} \;+\; \epsilon_{\text{exec}},$$
where $H \leq N$, $\epsilon_{\text{SRE}}$ is the SRE's held-out top-1 error, and $\epsilon_{\text{exec}}$ is the Action Decoder's error given the correct object. The two terms are additive, not multiplicative — neither source compounds the other. The **Heur** condition in Table 2 isolates $\epsilon_{\text{exec}}$ empirically by substituting Algorithm 1 for the SRE while keeping the Action Decoder unchanged.

---

### One-sentence "contribution" hook for the intro

> *We show that the decomposed architecture yields measurable learning benefits: object-centric inductive biases, a sample complexity governed only by the object-selection module, and an additive error structure that enables independent diagnosis of selection and execution failures.*

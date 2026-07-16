# URM MDN-Head Replacement: One-Page Design Summary

This document outlines the design decisions, mathematical rationale, and experimental results of replacing the single-Gaussian reward head in the Uncertainty-aware Reward Model (URM) with a Mixture Density Network (MDN) head.

---

### 1. Original URM Overview
The original URM (Uncertainty-aware Reward Model) utilizes a frozen **LLaMA-3.1-8B backbone** followed by a sequence classification head that outputs $10$ values (representing a mean $\mu$ and standard deviation $\sigma$ for $5$ human preference attributes: *helpfulness, correctness, coherence, complexity, verbosity*). An MLP gating network dynamically computes a set of $5$ weights based on the pooled hidden states, producing the final scalar reward score as the weighted sum of the expected attribute rewards.

---

### 2. Motivation for Replacing the Gaussian Head
Human evaluations of text quality attributes are often multimodal and asymmetric. For example, a response to a controversial prompt might receive highly positive ratings from one group of annotators and highly negative ratings from another, creating a bimodal rating distribution. Fitting a single Gaussian distribution to such ratings forces the model to average the modes and inflate its variance ($\sigma^2$), which leads to poor likelihood estimation (high NLL) and miscalibrated uncertainty scores.

---

### 3. Why a Mixture Density Network (MDN) Was Chosen
A Mixture Density Network (MDN) modeling a mixture of $K=3$ Logistic components per attribute ($45$ output dimensions total) resolves these limitations by:
* **Representing Multimodality**: Assigning separate means ($\mu_1, \mu_2, \mu_3$) to distinct clusters of human annotations.
* **Capturing Skewness and Heavy Tails**: Combining mixture weights ($\pi_1, \pi_2, \pi_3$) and heavy-tailed Logistic components to fit asymmetric, peaked empirical rating distributions without numeric instability.

---

### 4. Rationale for Freezing the Backbone
Fine-tuning an 8B parameter model is computationally expensive and risks catastrophic forgetting of the general language features learned during pre-training. Freezing the LLaMA backbone and training only the lightweight MDN head ($45 \times 4096 \approx 184\text{K}$ parameters) preserves the representation capacity of the model while allowing extremely fast optimization.

---

### 5. Rationale for Keeping the Gating Network Unchanged
To isolate the effect of the head replacement in our ablation study, the gating network architecture remains identical. This ensures that any change in performance, calibration, or uncertainty estimation is strictly attributable to the transition from a single-Gaussian representation to a Mixture Density representation.

---

### 6. Rationale for Using RM-Bench
`THU-KEG/RM-Bench` provides a high-quality, diverse set of pairwise preference evaluations spanning multiple categories. Evaluating both models on this dataset provides a standardized benchmark to measure how well the learned expected rewards translate to human-aligned pairwise preference classification.

---

### 7. Why NLL and Calibration Improved
The empirical distribution of human rating targets is highly non-Gaussian. The flexibility of the MDN allows it to allocate density directly to high-density rating zones while ignoring low-density gaps. This improved representation alignment directly reduced **Average NLL** from **1.823 to 1.452** and slashed **Calibration Error** by **61%** (from **0.124 to 0.048**).

---

### 8. Why Abstention Calibration Did Not Improve
While MDN-URM achieves a higher overall preference accuracy on RM-Bench (**54.63% vs 54.03%**), its **Abstention AUC** is slightly lower than the baseline (**0.5320 vs 0.5455**). 
* In the single-Gaussian URM baseline, the variance $\sigma^2$ directly tracks data mismatch and rating noise.
* In MDN-URM, the mixture variance $\text{Var}(r) = \sum_k \pi_k (\text{Var}_k + \mu_k^2) - \mathbb{E}[r]^2$ conflates two distinct sources of uncertainty: component-level scales ($s_k$, representing aleatoric noise) and mean separation ($\mu_k$, representing epistemic mode disagreement). This more complex uncertainty representation is less directly aligned with simple binary classification confidence, leading to a flatter abstention curve.

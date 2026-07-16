# Task: Implement MDN-URM Starting from the Official URM-LLaMA-3.1-8B Model
URM folder is the place where we implement 

## Goal

We want to build a new reward model by modifying the official URM implementation:

in the urm-LLama-3.1-8B folder.

The objective is to replace the current uncertainty-aware Gaussian attribute head with a Mixture Density Network (MDN) attribute head while keeping the rest of the URM pipeline as close as possible to the original.

This is an ablation study:

URM (single Gaussian per attribute)
vs
MDN-URM (mixture density per attribute)

The backbone should remain frozen.

---

# Step 1: Reverse Engineer URM

Inspect the model architecture and training pipeline.

Files to locate:

* modeling_custom.py
* training scripts
* HelpSteer2 preprocessing
* gating layer training code
* reward head implementation

Answer the following:

1. What is the exact shape of the current reward head output?
2. How are μ and σ represented?
3. How is σ constrained positive?
4. What loss is used during attribute regression?
5. Is reparameterization used?
6. How are the five attributes stored?
7. How is the gating network trained?
8. What exact BT loss implementation is used?
9. How are weights normalized?
10. Is softmax applied to gating outputs?

Create a markdown report:

docs/urm_reverse_engineering.md

with diagrams and code references.

---

# Step 2: Create MDN Head

Current URM outputs:

5 attributes × (μ, σ)

Total outputs = 10

Replace with:

5 attributes × K mixture components

Use:

K = 3

For each attribute output:

π1 π2 π3
μ1 μ2 μ3
σ1 σ2 σ3

Therefore:

Per attribute:

3 mixture weights
3 means
3 scales

Total = 9 values

For 5 attributes:

45 outputs

Implement:

models/mdn_head.py

Requirements:

* mixture weights via softmax
* scales via softplus + epsilon
* numerically stable logsumexp likelihood
* fp16 compatible

Create helper functions:

mdn_mean()
mdn_variance()
mdn_nll_loss()

---

# Step 3: Build MDN-URM Model

Create:

models/modeling_mdn_urm.py

Structure:

Frozen Llama Backbone
↓
MDN Attribute Head
↓
5 Attribute Scores
↓
Existing URM Gating Network
↓
Final Reward

For each attribute:

Expected reward:

E[r] = Σ π_k μ_k

Use expected reward as the attribute score fed into the gating network.

Keep the gating network unchanged initially.

---

# Step 4: Attribute Regression Training

Dataset:

nvidia/HelpSteer2

Train ONLY:

* MDN head

Freeze:

* Llama backbone

Use:

Negative Log Likelihood of the mixture

Target:

attribute scores from HelpSteer2

Create:

train_attribute_mdn.py

Features:

* gradient accumulation
* bf16 if available
* wandb logging
* checkpointing
* resume support
* deepspeed optional

Metrics:

* train loss
* validation NLL
* calibration plots
* attribute-wise RMSE

---

# Step 5: Gating Layer Training

Dataset:

Skywork/Skywork-Reward-Preference-80K-v0.1

Freeze:

* backbone
* MDN head

Train ONLY:

gating network

Use original URM BT loss.

Create:

train_gating.py

Output:

mdn_urm_gated_checkpoint

---

# Step 6: Evaluation

Implement:

evaluate_rewardbench.py

Metrics:

RewardBench accuracy

Attribute metrics

Uncertainty metrics

Create uncertainty-abstention curves:

retain %
vs
accuracy

Retain:

100%
95%
90%
85%
80%
75%
70%
60%
50%

Use MDN variance as uncertainty.

---

# Step 7: Comparison Against URM

Generate:

results/comparison_table.csv

Columns:

Model
RewardBench
Average NLL
Calibration Error
Abstention AUC

Rows:

URM
MDN-URM

Generate publication-ready plots.

---

# Step 8: SLURM Scripts

Create:

slurm/train_attribute_mdn.slurm

Requirements:

* A100/H100 compatible
* single-node
* 1, 2, 4 GPU options
* logs directory
* checkpoint directory
* automatic resume

Example:

#SBATCH --job-name=mdn_attr
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00

Launch:

accelerate launch train_attribute_mdn.py

---

Create:

slurm/train_gating.slurm

for stage 2.

---

# Step 9: Deliverables

At completion provide:

1. Architecture diagram.
2. Reverse-engineering report.
3. MDN implementation.
4. Training scripts.
5. Evaluation scripts.
6. SLURM scripts.
7. Exact commands to reproduce experiments.
8. Estimated GPU hours for:

   * Stage 1
   * Stage 2
   * RewardBench evaluation

Do not stop at planning.

Implement the code and provide all modified files.

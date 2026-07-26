"""
Run other diagnostic investigations requested by the user:
- 2. Residual uncertainty calibration (predicted SD/Var vs actual error/residual)
- 3. Residual distribution analysis (histogram & gaussianity check)
- 4. Effective number of active components (N_eff = exp(H))
- 5. Residual training scale check (mu and s vs actual residuals)
- 6. Loss balancing log analysis
- 7. Correlation of uncertainty with reward margin
"""
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import re
import math

from models.distribution_statistics import compute_statistics, mixture_mean, mixture_variance
from evaluation.scoring import get_urm_uncertainty, get_mad_uncertainty, get_urm_std

# Load caches
baseline = torch.load("results/baseline_predictions.pt", map_location="cpu")
residual = torch.load("results/residual_predictions.pt", map_location="cpu")

ATTRIBUTES = ["helpfulness", "correctness", "coherence", "complexity", "verbosity"]
out_dir = "results/analysis"
os.makedirs(out_dir, exist_ok=True)

# ─── 4. Effective number of active components (N_eff) ─────────────────────────
print("\n=== 4. Effective components (N_eff = exp(H)) ===")
# Let's extract this from the training log of Residual MDN (epoch 6)
# Helpful: H = 0.0238
# Correctness: H = 0.0275
# Coherence: H = 0.0630
# Complexity: H = 0.0156
# Verbosity: H = 0.0454
entropies = {
    "helpfulness": 0.0238,
    "correctness": 0.0275,
    "coherence": 0.0630,
    "complexity": 0.0156,
    "verbosity": 0.0454
}
for attr, H in entropies.items():
    n_eff = math.exp(H)
    print(f"  {attr:12s}: H={H:.4f} -> N_eff={n_eff:.4f} active components")

# ─── 6. Loss Balancing ────────────────────────────────────────────────────────
print("\n=== 6. Loss Balancing (MSE vs NLL) ===")
# Parse from results/slurm/mdn_residual_train_1074.out
log_path = "results/slurm/mdn_residual_train_1074.out"
if os.path.exists(log_path):
    with open(log_path) as f:
        log_content = f.read()
    
    losses = re.findall(r"Epoch (\d+) average training loss: ([\d.]+)", log_content)
    val_losses = re.findall(r"Epoch (\d+) validation joint loss: ([\d.]+)", log_content)
    
    # Let's also look at the loss components from training prints if any
    print(f"  Found {len(losses)} epochs of joint training loss:")
    for epoch, loss in losses:
        print(f"    Epoch {epoch}: joint loss = {loss}")
else:
    print("  Residual log not found.")

# ─── 7. Reward Margin vs Predicted Uncertainty ────────────────────────────────
print("\n=== 7. Reward Margin vs Predicted Uncertainty ===")
# Compute sequence reward margin
margin_base = (baseline["chosen_scores"] - baseline["rejected_scores"]).numpy()
margin_res = (residual["chosen_scores"] - residual["rejected_scores"]).numpy()

# Compute sequence uncertainty
ch_v_base = ((baseline["chosen_weights"] ** 2) * (baseline["chosen_sigmas"] ** 2)).sum(dim=-1)
rj_v_base = ((baseline["rejected_weights"] ** 2) * (baseline["rejected_sigmas"] ** 2)).sum(dim=-1)
unc_base = (ch_v_base + rj_v_base).numpy()

ch_res_stats = compute_statistics(residual["chosen_pi"], residual["chosen_mu"], residual["chosen_s"])
rj_res_stats = compute_statistics(residual["rejected_pi"], residual["rejected_mu"], residual["rejected_s"])
unc_res = (get_mad_uncertainty(ch_res_stats["mad0"], residual["chosen_weights"]) +
           get_mad_uncertainty(rj_res_stats["mad0"], residual["rejected_weights"])).numpy()

# Calculate Pearson correlation between absolute margin and uncertainty
corr_base = np.corrcoef(np.abs(margin_base), unc_base)[0, 1]
corr_res = np.corrcoef(np.abs(margin_res), unc_res)[0, 1]
print(f"  Gaussian URM: Correlation(|Margin|, Uncertainty) = {corr_base:.4f}")
print(f"  Residual MDN: Correlation(|Margin|, Uncertainty) = {corr_res:.4f}")

# Plot Margin vs Uncertainty
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.patch.set_facecolor("#0f1117")
for ax in axes:
    ax.set_facecolor("#1a1d27")
    ax.tick_params(colors="#aaaaaa")
    for sp in ax.spines.values():
        sp.set_edgecolor("#333344")
    ax.grid(color="#2a2d3a", linewidth=0.8, alpha=0.5)

axes[0].scatter(margin_base, unc_base, alpha=0.3, color="#64b5f6", s=8)
axes[0].set_title(f"Gaussian URM\nCorr(|Margin|, Unc) = {corr_base:.3f}", color="white")
axes[0].set_xlabel("Reward Margin (Chosen - Rejected)", color="#cccccc")
axes[0].set_ylabel("Pairwise Uncertainty (Var)", color="#cccccc")

axes[1].scatter(margin_res, unc_res, alpha=0.3, color="#a5d6a7", s=8)
axes[1].set_title(f"Residual MDN\nCorr(|Margin|, Unc) = {corr_res:.3f}", color="white")
axes[1].set_xlabel("Reward Margin (Chosen - Rejected)", color="#cccccc")
axes[1].set_ylabel("Pairwise Uncertainty (MAD)", color="#cccccc")

plt.tight_layout()
margin_vs_unc_path = os.path.join(out_dir, "margin_vs_uncertainty_scatter.png")
plt.savefig(margin_vs_unc_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"  Saved plot: {margin_vs_unc_path}")
plt.close()

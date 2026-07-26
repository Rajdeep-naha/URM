"""
Analyze the abstention dip at the sample level.
This script compares Gaussian URM and Residual MDN, finding exactly which samples
are removed (abstained on) at each retention step, and saving their metadata,
rewards, margins, and prediction details.
"""
import torch
import numpy as np
import os
import csv
import math
import torch.nn.functional as F

from models.distribution_statistics import compute_statistics
from evaluation.scoring import get_urm_uncertainty, get_mad_uncertainty, get_urm_std

# Load predictions
baseline = torch.load("results/baseline_predictions.pt", map_location="cpu")
residual = torch.load("results/residual_predictions.pt", map_location="cpu")

N = len(baseline["chosen_scores"])
print(f"Total samples loaded: {N}")

# 1. Compute uncertainty for Gaussian URM
correct_base = (baseline["chosen_scores"] > baseline["rejected_scores"]).numpy().astype(int)
ch_v_base = ((baseline["chosen_weights"] ** 2) * (baseline["chosen_sigmas"] ** 2)).sum(dim=-1)
rj_v_base = ((baseline["rejected_weights"] ** 2) * (baseline["rejected_sigmas"] ** 2)).sum(dim=-1)
unc_base = (ch_v_base + rj_v_base).numpy()
margin_base = (baseline["chosen_scores"] - baseline["rejected_scores"]).numpy()

# 2. Compute uncertainty for Residual MDN (MAD)
correct_res = (residual["chosen_scores"] > residual["rejected_scores"]).numpy().astype(int)
ch_res_stats = compute_statistics(residual["chosen_pi"], residual["chosen_mu"], residual["chosen_s"])
rj_res_stats = compute_statistics(residual["rejected_pi"], residual["rejected_mu"], residual["rejected_s"])
unc_res = (get_mad_uncertainty(ch_res_stats["mad0"], residual["chosen_weights"]) +
           get_mad_uncertainty(rj_res_stats["mad0"], residual["rejected_weights"])).numpy()
margin_res = (residual["chosen_scores"] - residual["rejected_scores"]).numpy()

# Define retention steps
steps = [1.0, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.60, 0.50]

def get_retained_indices(unc, retain_rate):
    cutoff = np.percentile(unc, retain_rate * 100)
    retained_idx = np.where(unc <= cutoff)[0]
    return set(retained_idx)

# Find removed indices at each transition
transitions = [(steps[i], steps[i+1]) for i in range(len(steps)-1)]

os.makedirs("results/analysis", exist_ok=True)
csv_out = "results/analysis/abstention_transitions.csv"

with open(csv_out, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "Model", "Transition", "Idx", "Correct", "Margin", "Uncertainty", 
        "Category", "Subcategory", "Prompt", "Chosen", "Rejected"
    ])
    
    for start, end in transitions:
        # Gaussian URM
        base_retained_start = get_retained_indices(unc_base, start)
        base_retained_end = get_retained_indices(unc_base, end)
        base_removed = base_retained_start - base_retained_end
        
        # Calculate accuracy of removed subset
        if len(base_removed) > 0:
            base_removed_correct = [correct_base[idx] for idx in base_removed]
            acc_removed = np.mean(base_removed_correct)
            print(f"Gaussian URM {start*100:.0f}%->{end*100:.0f}%: Removed {len(base_removed)} samples. Accuracy of removed: {acc_removed:.4f}")
            for idx in base_removed:
                writer.writerow([
                    "Gaussian URM", f"{start*100:.0f}%->{end*100:.0f}%", idx, correct_base[idx],
                    margin_base[idx], unc_base[idx], baseline["categories"][idx], baseline["subcategories"][idx],
                    baseline["prompts"][idx][:200] + "...", baseline["chosen_texts"][idx][:100] + "...", baseline["rejected_texts"][idx][:100] + "..."
                ])
                
        # Residual MDN
        res_retained_start = get_retained_indices(unc_res, start)
        res_retained_end = get_retained_indices(unc_res, end)
        res_removed = res_retained_start - res_retained_end
        
        if len(res_removed) > 0:
            res_removed_correct = [correct_res[idx] for idx in res_removed]
            acc_removed_res = np.mean(res_removed_correct)
            print(f"Residual MDN {start*100:.0f}%->{end*100:.0f}%: Removed {len(res_removed)} samples. Accuracy of removed: {acc_removed_res:.4f}")
            for idx in res_removed:
                writer.writerow([
                    "Residual MDN", f"{start*100:.0f}%->{end*100:.0f}%", idx, correct_res[idx],
                    margin_res[idx], unc_res[idx], residual["categories"][idx], residual["subcategories"][idx],
                    residual["prompts"][idx][:200] + "...", residual["chosen_texts"][idx][:100] + "...", residual["rejected_texts"][idx][:100] + "..."
                ])

print(f"Transition details saved to {csv_out}")

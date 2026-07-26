import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import math

from models.distribution_statistics import compute_statistics

ATTRIBUTES = ["helpfulness", "correctness", "coherence", "complexity", "verbosity"]

def get_abstention_curve(scores_c, scores_r, unc, steps=20):
    correct = (scores_c > scores_r).numpy().astype(int)
    unc = unc.numpy()
    
    # Sort by uncertainty (ascending)
    sorted_indices = np.argsort(unc)
    correct_sorted = correct[sorted_indices]
    
    # Calculate accuracy at different retention rates
    retention_rates = np.linspace(1.0, 0.1, steps)
    accuracies = []
    
    for rate in retention_rates:
        num_keep = max(1, int(len(correct_sorted) * rate))
        acc = np.mean(correct_sorted[:num_keep])
        accuracies.append(acc)
        
    return retention_rates, accuracies

def process_model_cache(cache_path, is_mdn):
    if not os.path.exists(cache_path):
        return None
        
    data = torch.load(cache_path, map_location="cpu")
    
    # Check if this cache is fully ready (baseline needs chosen_mu)
    if not is_mdn and "chosen_mu" not in data:
        print(f"Warning: {cache_path} does not contain 'chosen_mu'. Wait for cache regeneration.")
        return None
        
    if not is_mdn and len(data["chosen_mu"]) == 0:
        print(f"Warning: {cache_path} 'chosen_mu' is empty. Wait for cache regeneration.")
        return None

    results = {}
    
    for i, attr in enumerate(ATTRIBUTES):
        if is_mdn:
            # MDN parameters
            pi_c = data["chosen_pi"]
            mu_c = data["chosen_mu"]
            s_c = data["chosen_s"]
            
            pi_r = data["rejected_pi"]
            mu_r = data["rejected_mu"]
            s_r = data["rejected_s"]
            
            # Expected Reward = sum(pi * mu)
            scores_c = torch.sum(pi_c[:, i, :] * mu_c[:, i, :], dim=-1)
            scores_r = torch.sum(pi_r[:, i, :] * mu_r[:, i, :], dim=-1)
            
            # Uncertainty = MAD
            ch_stats = compute_statistics(pi_c[:, i, :], mu_c[:, i, :], s_c[:, i, :])
            rj_stats = compute_statistics(pi_r[:, i, :], mu_r[:, i, :], s_r[:, i, :])
            
            unc = ch_stats["mad0"] + rj_stats["mad0"]
            
        else:
            # Gaussian URM
            # Expected Reward = mu
            mu_c = data["chosen_mu"]
            mu_r = data["rejected_mu"]
            
            sigma_c = data["chosen_sigmas"]
            sigma_r = data["rejected_sigmas"]
            
            scores_c = mu_c[:, i]
            scores_r = mu_r[:, i]
            
            # Uncertainty = Variance
            unc = (sigma_c[:, i]**2) + (sigma_r[:, i]**2)
            
        retention, acc = get_abstention_curve(scores_c, scores_r, unc)
        base_acc = acc[0]
        
        results[attr] = {
            "retention": retention,
            "accuracy": acc,
            "base_accuracy": base_acc
        }
        
    return results

def main():
    os.makedirs("results/analysis", exist_ok=True)
    
    models = {
        # "Gaussian URM": ("results/baseline_predictions.pt", False, "blue", "-"),
        "Label MDN": ("results/label_predictions.pt", True, "orange", "--"),
        "Residual MDN": ("results/residual_predictions.pt", True, "green", "-")
    }
    
    model_results = {}
    for name, (path, is_mdn, color, style) in models.items():
        res = process_model_cache(path, is_mdn)
        if res is not None:
            model_results[name] = (res, color, style)
            
    if not model_results:
        print("No valid caches found.")
        return

    # Plot abstention curves for each attribute
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    fig.patch.set_facecolor('#0f1117')
    
    for i, attr in enumerate(ATTRIBUTES):
        ax = axes[i]
        ax.set_facecolor('#1a1d27')
        ax.tick_params(colors='#aaaaaa')
        for sp in ax.spines.values():
            sp.set_edgecolor('#333344')
        ax.grid(color='#2a2d3a', linewidth=0.8, alpha=0.5)
        
        for name, (res, color, style) in model_results.items():
            attr_res = res[attr]
            ax.plot(attr_res["retention"] * 100, attr_res["accuracy"], 
                    label=f'{name} (Base: {attr_res["base_accuracy"]:.3f})', 
                    color=color, linestyle=style, linewidth=2)
            
        ax.invert_xaxis()  # 100% to 10%
        ax.set_title(f"{attr.capitalize()} Abstention", color='white', fontsize=12)
        ax.set_xlabel("Retention Rate (%)", color='#cccccc')
        if i == 0:
            ax.set_ylabel("Accuracy", color='#cccccc')
        ax.legend(facecolor='#1a1d27', edgecolor='#333344', labelcolor='white', fontsize=9)
        
    plt.tight_layout()
    out_path = "results/analysis/single_attribute_abstention.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved plot to {out_path}")

if __name__ == "__main__":
    main()

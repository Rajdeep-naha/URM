"""
Fair Single-Attribute Evaluation on HelpSteer2 Validation Set.

Three evaluation axes, each using the appropriate metric:

1. PREDICTION QUALITY (Label MDN & Residual MDN only)
   - These models were trained on HelpSteer2 labels, so MAE/RMSE is meaningful
   - Uses `expected_attribute_rewards` (outputs[2]) denormalized to original scale
   - The Gaussian URM baseline was NOT trained on HelpSteer2, so it cannot be compared

2. UNCERTAINTY QUALITY (all three models)
   - Abstention: sort by uncertainty, remove most uncertain → does error/accuracy improve?
   - For Gaussian URM: uncertainty = sigma from the pretrained model
   - For Label/Residual MDN: uncertainty = MAD of the mixture

3. RANKING QUALITY (all three models)
   - Given pairs of samples with different labels, can the model rank them correctly?
   - This is scale-invariant, so it's fair for all models
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from datasets import load_dataset
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from models.modeling_mdn_urm import LlamaForSequenceClassificationWithMDN
from models.distribution_statistics import compute_statistics

ATTRIBUTES = ["helpfulness", "correctness", "coherence", "complexity", "verbosity"]


class HelpSteer2Dataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item["prompt"]
        response = item["response"]
        conversation = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response}
        ]
        text = self.tokenizer.apply_chat_template(conversation, tokenize=False)
        inputs = self.tokenizer(text, max_length=self.max_length, padding="max_length",
                                truncation=True, return_tensors="pt")
        labels = torch.tensor([float(item[attr]) for attr in ATTRIBUTES], dtype=torch.float32)
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": labels,
        }


def compute_normalization_stats(train_data):
    attr_means, attr_stds = [], []
    for attr in ATTRIBUTES:
        vals = [float(x[attr]) for x in train_data]
        mean, std = np.mean(vals), np.std(vals)
        if std < 1e-6: std = 1.0
        attr_means.append(mean)
        attr_stds.append(std)
    return torch.tensor(attr_means, dtype=torch.float32), torch.tensor(attr_stds, dtype=torch.float32)


def remove_hooks_and_materialize(model, device):
    from accelerate.hooks import remove_hook_from_module
    for module_name in ["score", "weights"]:
        if not hasattr(model, module_name): continue
        module = getattr(model, module_name)
        remove_hook_from_module(module, recurse=True)
        def materialize_submodule(submod):
            for param_name, param in list(submod.named_parameters(recurse=False)):
                if param.device.type == "meta":
                    new_param = nn.Parameter(torch.empty_like(param, device=device))
                    if "weight" in param_name: nn.init.xavier_uniform_(new_param)
                    else: nn.init.zeros_(new_param)
                    submod.register_parameter(param_name, new_param)
            for child in submod.children(): materialize_submodule(child)
        materialize_submodule(module)
        module.to(device)


def evaluate_mdn_model(model, val_loader, device, attr_means, attr_stds, uncertainty_target):
    """
    Evaluate an MDN model. Returns per-sample predictions (original scale) and uncertainties.
    Uses outputs[2] (expected_attribute_rewards) which is the proper prediction for both
    label and residual modes.
    """
    model.eval()
    all_labels, all_preds, all_uncs = [], [], []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Evaluating ({uncertainty_target})"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_orig = batch["labels"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
            # outputs[2] = expected_attribute_rewards (in normalized space)
            expected_rewards_norm = outputs[2].cpu().to(torch.float32)  # [B, 5]
            
            # Denormalize to original label scale (0-4)
            pred_orig = expected_rewards_norm * attr_stds + attr_means

            # Get uncertainty from the distribution parameters
            params = outputs[3]
            pi, mu, s = params
            pi = pi.cpu().to(torch.float32)
            mu = mu.cpu().to(torch.float32)
            s = s.cpu().to(torch.float32)
            stats = compute_statistics(pi, mu, s)
            # MAD in normalized space → scale to original
            unc_orig = stats["mad0"] * attr_stds

            all_labels.append(labels_orig)
            all_preds.append(pred_orig)
            all_uncs.append(unc_orig)

    return {
        "labels_orig": torch.cat(all_labels, dim=0),
        "predictions_orig": torch.cat(all_preds, dim=0),
        "uncertainties": torch.cat(all_uncs, dim=0),
    }


def evaluate_gaussian_baseline(model, val_loader, device):
    """
    Evaluate the pretrained Gaussian URM baseline.
    Its raw mu values are NOT on the label scale, so we store them as-is
    for ranking comparisons only.
    """
    model.eval()
    all_labels, all_preds, all_uncs = [], [], []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating (Gaussian URM)"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_orig = batch["labels"]

            pooled_logits, weights = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
            params = pooled_logits.view(-1, 5, 2).cpu().to(torch.float32)
            mu = params[:, :, 0]       # [B, 5] - internal scale, NOT label scale
            sigma = F.softplus(params[:, :, 1]) + 1e-6  # [B, 5]

            all_labels.append(labels_orig)
            all_preds.append(mu)        # Raw predictions (internal scale)
            all_uncs.append(sigma)      # Raw uncertainty

    return {
        "labels_orig": torch.cat(all_labels, dim=0),
        "predictions_orig": torch.cat(all_preds, dim=0),  # NOTE: not on label scale!
        "uncertainties": torch.cat(all_uncs, dim=0),
    }


def pairwise_ranking_accuracy(predictions, labels, attr_idx):
    """
    For a given attribute, sample random pairs and check if the model
    ranks them in the same order as the ground-truth labels.
    Scale-invariant — only cares about ordering.
    """
    N = len(predictions)
    n_pairs = min(50000, N * (N - 1) // 2)
    
    np.random.seed(42)
    idx_a = np.random.randint(0, N, size=n_pairs)
    idx_b = np.random.randint(0, N, size=n_pairs)
    # Remove self-pairs and ties
    valid = (idx_a != idx_b) & (labels[idx_a, attr_idx].numpy() != labels[idx_b, attr_idx].numpy())
    idx_a, idx_b = idx_a[valid], idx_b[valid]

    label_order = labels[idx_a, attr_idx] > labels[idx_b, attr_idx]
    pred_order = predictions[idx_a, attr_idx] > predictions[idx_b, attr_idx]
    
    return (label_order.numpy() == pred_order.numpy()).mean()


def plot_all(results, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    
    colors = {"Gaussian URM": "#4a9eff", "Label MDN": "#ff8c00", "Residual MDN": "#2ecc71"}
    
    # Separate MDN models (which have predictions on label scale) from baseline
    mdn_models = {k: v for k, v in results.items() if k != "Gaussian URM"}
    
    # =========================================================================
    # PLOT 1: Prediction MAE (MDN models only — Gaussian URM is not on label scale)
    # =========================================================================
    if mdn_models:
        fig, axes = plt.subplots(1, 5, figsize=(28, 5.5))
        fig.patch.set_facecolor('#0f1117')
        fig.suptitle("Abstention on Prediction Error — HelpSteer2 Val\n(Label MDN & Residual MDN only — Gaussian URM not trained on labels)",
                     color="white", fontsize=14, y=1.04)

        retention_rates = np.linspace(1.0, 0.1, 20)

        for i, attr in enumerate(ATTRIBUTES):
            ax = axes[i]
            ax.set_facecolor('#1a1d27')
            ax.tick_params(colors='#aaaaaa')
            for sp in ax.spines.values(): sp.set_edgecolor('#333344')
            ax.grid(color='#2a2d3a', linewidth=0.8, alpha=0.5)

            for name, res in mdn_models.items():
                errors = np.abs((res["predictions_orig"][:, i] - res["labels_orig"][:, i]).numpy())
                unc = res["uncertainties"][:, i].numpy()
                sorted_idx = np.argsort(unc)
                errors_sorted = errors[sorted_idx]

                maes = []
                for rate in retention_rates:
                    n_keep = max(1, int(len(errors_sorted) * rate))
                    maes.append(np.mean(errors_sorted[:n_keep]))

                ax.plot(retention_rates * 100, maes, label=f"{name} (MAE@100%: {maes[0]:.3f})",
                        color=colors[name], linewidth=2.5)

            ax.invert_xaxis()
            ax.set_title(f"{attr.capitalize()}", color="white", fontsize=13, fontweight="bold")
            ax.set_xlabel("Retention Rate (%)", color="#cccccc")
            if i == 0: ax.set_ylabel("MAE", color="#cccccc", fontsize=11)
            ax.legend(facecolor='#1a1d27', edgecolor='#333344', labelcolor='white', fontsize=9)

        plt.tight_layout()
        path = os.path.join(save_dir, "abstention_on_error_mdn.png")
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()
        print(f"Saved: {path}")

    # =========================================================================
    # PLOT 2: Pairwise Ranking Accuracy (ALL models — scale invariant)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('#0f1117')
    ax.set_facecolor('#1a1d27')
    ax.tick_params(colors='#aaaaaa')
    for sp in ax.spines.values(): sp.set_edgecolor('#333344')
    ax.grid(color='#2a2d3a', linewidth=0.8, alpha=0.5, axis='y')

    x = np.arange(len(ATTRIBUTES))
    width = 0.25
    model_names = list(results.keys())

    print("\n" + "=" * 70)
    print(f"{'Attribute':<15}", end="")
    for name in model_names:
        print(f"  {name:>16}", end="")
    print("\n" + "-" * 70)

    for j, name in enumerate(model_names):
        accs = []
        for i, attr in enumerate(ATTRIBUTES):
            acc = pairwise_ranking_accuracy(results[name]["predictions_orig"],
                                           results[name]["labels_orig"], i)
            accs.append(acc)
        
        # Print table row
        if j == 0:
            for i, attr in enumerate(ATTRIBUTES):
                line = f"{attr:<15}"
                for jj, nm in enumerate(model_names):
                    a = pairwise_ranking_accuracy(results[nm]["predictions_orig"],
                                                  results[nm]["labels_orig"], i)
                    line += f"  {a:>16.4f}"
                print(line)

        ax.bar(x + j * width, accs, width, label=name, color=colors[name], alpha=0.85)

    print("=" * 70)

    ax.set_xticks(x + width)
    ax.set_xticklabels([a.capitalize() for a in ATTRIBUTES], color="#cccccc", fontsize=12)
    ax.set_ylabel("Pairwise Ranking Accuracy", color="#cccccc", fontsize=12)
    ax.set_title("Pairwise Ranking Accuracy per Attribute (Scale-Invariant)\nDoes the model rank two samples in the same order as ground truth?",
                 color="white", fontsize=14)
    ax.legend(facecolor='#1a1d27', edgecolor='#333344', labelcolor='white', fontsize=11)
    ax.axhline(y=0.5, color='#666666', linestyle='--', linewidth=1, alpha=0.7, label="Random")
    ax.set_ylim(0.4, 1.0)

    path = os.path.join(save_dir, "pairwise_ranking_accuracy.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {path}")

    # =========================================================================
    # PLOT 3: Uncertainty-Error Correlation (ALL models)
    #   For Gaussian URM: error = |pred_rank - label_rank| (rank-based)
    #   For MDN models: error = |pred - label| (actual error)
    # =========================================================================
    fig, axes = plt.subplots(1, 5, figsize=(28, 5.5))
    fig.patch.set_facecolor('#0f1117')
    fig.suptitle("Uncertainty vs. Prediction Error — HelpSteer2 Val", color="white", fontsize=14, y=1.02)

    for i, attr in enumerate(ATTRIBUTES):
        ax = axes[i]
        ax.set_facecolor('#1a1d27')
        ax.tick_params(colors='#aaaaaa')
        for sp in ax.spines.values(): sp.set_edgecolor('#333344')
        ax.grid(color='#2a2d3a', linewidth=0.8, alpha=0.5)

        legend_entries = []
        for name, res in results.items():
            if name == "Gaussian URM":
                # For Gaussian: use rank-based error (percentile of pred vs label)
                pred_ranks = torch.argsort(torch.argsort(res["predictions_orig"][:, i])).float()
                label_ranks = torch.argsort(torch.argsort(res["labels_orig"][:, i])).float()
                errors = np.abs((pred_ranks - label_ranks).numpy()) / len(pred_ranks)
            else:
                errors = np.abs((res["predictions_orig"][:, i] - res["labels_orig"][:, i]).numpy())
            
            unc = res["uncertainties"][:, i].numpy()
            corr = np.corrcoef(unc, errors)[0, 1]
            ax.scatter(unc, errors, alpha=0.12, s=6, color=colors[name], edgecolors="none")
            legend_entries.append(f"{name} (ρ={corr:.3f})")

        ax.set_title(f"{attr.capitalize()}", color="white", fontsize=13, fontweight="bold")
        ax.set_xlabel("Predicted Uncertainty", color="#cccccc")
        if i == 0: ax.set_ylabel("Error", color="#cccccc")
        ax.legend(legend_entries, facecolor='#1a1d27', edgecolor='#333344', labelcolor='white', fontsize=8)

    plt.tight_layout()
    path = os.path.join(save_dir, "uncertainty_vs_error_fair.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {path}")

    # =========================================================================
    # PLOT 4: Calibration (MDN models only)
    # =========================================================================
    if mdn_models:
        fig, axes = plt.subplots(1, 5, figsize=(28, 5.5))
        fig.patch.set_facecolor('#0f1117')
        fig.suptitle("Calibration: Predicted Uncertainty vs. Actual RMSE", color="white", fontsize=14, y=1.02)

        n_bins = 10
        for i, attr in enumerate(ATTRIBUTES):
            ax = axes[i]
            ax.set_facecolor('#1a1d27')
            ax.tick_params(colors='#aaaaaa')
            for sp in ax.spines.values(): sp.set_edgecolor('#333344')
            ax.grid(color='#2a2d3a', linewidth=0.8, alpha=0.5)

            for name, res in mdn_models.items():
                errors = (res["predictions_orig"][:, i] - res["labels_orig"][:, i]).numpy()
                unc = res["uncertainties"][:, i].numpy()
                sorted_idx = np.argsort(unc)
                bin_size = len(sorted_idx) // n_bins
                bin_uncs, bin_rmses = [], []
                for b in range(n_bins):
                    start = b * bin_size
                    end = start + bin_size if b < n_bins - 1 else len(sorted_idx)
                    idx = sorted_idx[start:end]
                    bin_uncs.append(np.mean(unc[idx]))
                    bin_rmses.append(np.sqrt(np.mean(errors[idx] ** 2)))
                ax.plot(bin_uncs, bin_rmses, marker="o", markersize=5, label=name,
                        color=colors[name], linewidth=2)

            max_val = max(max(res["uncertainties"][:, i].numpy()) for res in mdn_models.values())
            ax.plot([0, max_val], [0, max_val], "--", color="#666666", linewidth=1, alpha=0.7)
            ax.set_title(f"{attr.capitalize()}", color="white", fontsize=13, fontweight="bold")
            ax.set_xlabel("Binned Predicted Uncertainty", color="#cccccc")
            if i == 0: ax.set_ylabel("Binned Actual RMSE", color="#cccccc")
            ax.legend(facecolor='#1a1d27', edgecolor='#333344', labelcolor='white', fontsize=9)

        plt.tight_layout()
        path = os.path.join(save_dir, "calibration_fair.png")
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()
        print(f"Saved: {path}")

    # =========================================================================
    # PLOT 5: Summary — MAE (MDN only) + Ranking Accuracy (all) + Correlation (all)
    # =========================================================================
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 6))
    fig.patch.set_facecolor('#0f1117')

    for ax in [ax1, ax2, ax3]:
        ax.set_facecolor('#1a1d27')
        ax.tick_params(colors='#aaaaaa')
        for sp in ax.spines.values(): sp.set_edgecolor('#333344')
        ax.grid(color='#2a2d3a', linewidth=0.8, alpha=0.5, axis='y')

    x = np.arange(len(ATTRIBUTES))
    width = 0.25

    # Panel 1: MAE (MDN models only)
    mdn_names = list(mdn_models.keys())
    for j, name in enumerate(mdn_names):
        maes = [np.mean(np.abs((results[name]["predictions_orig"][:, i] - results[name]["labels_orig"][:, i]).numpy()))
                for i in range(5)]
        ax1.bar(x + j * width, maes, width, label=name, color=colors[name], alpha=0.85)
    ax1.set_xticks(x + width * (len(mdn_names) - 1) / 2)
    ax1.set_xticklabels([a.capitalize() for a in ATTRIBUTES], color="#cccccc")
    ax1.set_ylabel("MAE (original scale)", color="#cccccc")
    ax1.set_title("Prediction MAE\n(MDN models only)", color="white", fontsize=13)
    ax1.legend(facecolor='#1a1d27', edgecolor='#333344', labelcolor='white')

    # Panel 2: Pairwise Ranking Accuracy (all models)
    all_names = list(results.keys())
    for j, name in enumerate(all_names):
        accs = [pairwise_ranking_accuracy(results[name]["predictions_orig"],
                                          results[name]["labels_orig"], i) for i in range(5)]
        ax2.bar(x + j * width, accs, width, label=name, color=colors[name], alpha=0.85)
    ax2.set_xticks(x + width)
    ax2.set_xticklabels([a.capitalize() for a in ATTRIBUTES], color="#cccccc")
    ax2.set_ylabel("Pairwise Accuracy", color="#cccccc")
    ax2.set_title("Ranking Accuracy\n(all models, scale-invariant)", color="white", fontsize=13)
    ax2.legend(facecolor='#1a1d27', edgecolor='#333344', labelcolor='white')
    ax2.axhline(y=0.5, color='#666', linestyle='--', alpha=0.5)

    # Panel 3: Uncertainty-Error Correlation (all models)
    for j, name in enumerate(all_names):
        corrs = []
        for i in range(5):
            if name == "Gaussian URM":
                pred_ranks = torch.argsort(torch.argsort(results[name]["predictions_orig"][:, i])).float()
                label_ranks = torch.argsort(torch.argsort(results[name]["labels_orig"][:, i])).float()
                errors = np.abs((pred_ranks - label_ranks).numpy()) / len(pred_ranks)
            else:
                errors = np.abs((results[name]["predictions_orig"][:, i] - results[name]["labels_orig"][:, i]).numpy())
            unc = results[name]["uncertainties"][:, i].numpy()
            corrs.append(np.corrcoef(unc, errors)[0, 1])
        ax3.bar(x + j * width, corrs, width, label=name, color=colors[name], alpha=0.85)
    ax3.set_xticks(x + width)
    ax3.set_xticklabels([a.capitalize() for a in ATTRIBUTES], color="#cccccc")
    ax3.set_ylabel("Pearson ρ (Unc. vs Error)", color="#cccccc")
    ax3.set_title("Uncertainty-Error Correlation\n(all models)", color="white", fontsize=13)
    ax3.legend(facecolor='#1a1d27', edgecolor='#333344', labelcolor='white')

    plt.tight_layout()
    path = os.path.join(save_dir, "summary_fair.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="/localstorage/home/f20221218/URM-LLaMa-3.1-8B")
    parser.add_argument("--label_mdn_weights", type=str, default="checkpoints/stage1/best_mdn_head_label.pt")
    parser.add_argument("--residual_mdn_weights", type=str, default="checkpoints/stage1/best_mdn_head_residual.pt")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--save_dir", type=str, default="results/helpsteer2_eval")
    parser.add_argument("--num_components", type=int, default=3)
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--cache_dir", type=str, default="results/helpsteer2_eval/caches")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading HelpSteer2...")
    dataset = load_dataset("nvidia/HelpSteer2")
    train_data, val_data = dataset["train"], dataset["validation"]

    attr_means, attr_stds = compute_normalization_stats(train_data)
    print(f"Normalization: means={attr_means.tolist()}, stds={attr_stds.tolist()}")

    val_dataset = HelpSteer2Dataset(val_data, tokenizer, max_length=args.max_length)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    import gc
    results = {}

    # === 1. Gaussian URM Baseline ===
    cache_path = os.path.join(args.cache_dir, "gaussian_helpsteer2_v2.pt")
    if os.path.exists(cache_path):
        print(f"\nLoading cached Gaussian URM from {cache_path}")
        results["Gaussian URM"] = torch.load(cache_path, map_location="cpu")
    else:
        print("\n=== Evaluating Gaussian URM Baseline ===")
        from transformers import AutoModelForSequenceClassification
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name_or_path, trust_remote_code=True,
            torch_dtype=torch.bfloat16, device_map=args.device_map)
        results["Gaussian URM"] = evaluate_gaussian_baseline(model, val_loader, device)
        torch.save(results["Gaussian URM"], cache_path)
        del model; gc.collect(); torch.cuda.empty_cache()

    # === 2. Label MDN ===
    cache_path = os.path.join(args.cache_dir, "label_mdn_helpsteer2_v2.pt")
    if os.path.exists(cache_path):
        print(f"\nLoading cached Label MDN from {cache_path}")
        results["Label MDN"] = torch.load(cache_path, map_location="cpu")
    elif os.path.exists(args.label_mdn_weights):
        print("\n=== Evaluating Label MDN ===")
        model = LlamaForSequenceClassificationWithMDN.from_pretrained(
            args.model_name_or_path, ignore_mismatched_sizes=True,
            torch_dtype=torch.bfloat16, device_map=args.device_map,
            num_components=args.num_components, uncertainty_target="label")
        if model.score.proj.weight.device.type == "meta":
            remove_hooks_and_materialize(model, device)
        model.score.load_state_dict(torch.load(args.label_mdn_weights, map_location="cpu"))
        model.score.to(torch.bfloat16)
        results["Label MDN"] = evaluate_mdn_model(model, val_loader, device, attr_means, attr_stds, "label")
        torch.save(results["Label MDN"], cache_path)
        del model; gc.collect(); torch.cuda.empty_cache()

    # === 3. Residual MDN ===
    cache_path = os.path.join(args.cache_dir, "residual_mdn_helpsteer2_v2.pt")
    if os.path.exists(cache_path):
        print(f"\nLoading cached Residual MDN from {cache_path}")
        results["Residual MDN"] = torch.load(cache_path, map_location="cpu")
    elif os.path.exists(args.residual_mdn_weights):
        print("\n=== Evaluating Residual MDN ===")
        model = LlamaForSequenceClassificationWithMDN.from_pretrained(
            args.model_name_or_path, ignore_mismatched_sizes=True,
            torch_dtype=torch.bfloat16, device_map=args.device_map,
            num_components=args.num_components, uncertainty_target="residual")
        if model.score.proj.weight.device.type == "meta":
            remove_hooks_and_materialize(model, device)
        model.score.load_state_dict(torch.load(args.residual_mdn_weights, map_location="cpu"))
        model.score.to(torch.bfloat16)
        results["Residual MDN"] = evaluate_mdn_model(model, val_loader, device, attr_means, attr_stds, "residual")
        torch.save(results["Residual MDN"], cache_path)
        del model; gc.collect(); torch.cuda.empty_cache()

    # === Generate Plots ===
    if results:
        plot_all(results, args.save_dir)
        
        # Print MAE table for MDN models
        print("\n" + "=" * 50)
        print("PREDICTION MAE (MDN models only, on label scale 0-4)")
        print("-" * 50)
        mdn_models = {k: v for k, v in results.items() if k != "Gaussian URM"}
        print(f"{'Attribute':<15}", end="")
        for name in mdn_models: print(f"  {name:>14}", end="")
        print()
        for i, attr in enumerate(ATTRIBUTES):
            line = f"{attr:<15}"
            for name in mdn_models:
                mae = np.mean(np.abs((results[name]["predictions_orig"][:, i] - results[name]["labels_orig"][:, i]).numpy()))
                line += f"  {mae:>14.4f}"
            print(line)
        print("=" * 50)
        print("\nDone!")


if __name__ == "__main__":
    main()

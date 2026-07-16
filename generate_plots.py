import os
import gc
import csv
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from models.modeling_mdn_urm import LlamaForSequenceClassificationWithMDN
from models.mdn_head import mixture_variance, mixture_mean

# Configuration
MODEL_PATH = "/localstorage/home/f20221218/URM-LLaMa-3.1-8B"
MDN_HEAD_WEIGHTS = "checkpoints/stage1/best_mdn_head.pt"
GATING_WEIGHTS = "checkpoints/stage2/best_gating_weights.pt"
MAX_LENGTH = 512
BATCH_SIZE = 4

class EvalPreferenceDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item["prompt"]
        chosen = item["chosen"]
        rejected = item["rejected"]

        chosen_conv = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": chosen}
        ]
        chosen_text = self.tokenizer.apply_chat_template(chosen_conv, tokenize=False)

        rejected_conv = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": rejected}
        ]
        rejected_text = self.tokenizer.apply_chat_template(rejected_conv, tokenize=False)

        chosen_inputs = self.tokenizer(
            chosen_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        rejected_inputs = self.tokenizer(
            rejected_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "chosen_input_ids": chosen_inputs["input_ids"].squeeze(0),
            "chosen_attention_mask": chosen_inputs["attention_mask"].squeeze(0),
            "rejected_input_ids": rejected_inputs["input_ids"].squeeze(0),
            "rejected_attention_mask": rejected_inputs["attention_mask"].squeeze(0),
        }

def remove_hooks_and_materialize_meta_parameters(model, device):
    import torch.nn as nn
    from accelerate.hooks import remove_hook_from_module
    for module_name in ["score", "weights"]:
        if not hasattr(model, module_name):
            continue
        module = getattr(model, module_name)
        remove_hook_from_module(module, recurse=True)
        def materialize_submodule(submod):
            for param_name, param in list(submod.named_parameters(recurse=False)):
                if param.device.type == "meta":
                    new_param = nn.Parameter(torch.empty_like(param, device=device))
                    if "weight" in param_name:
                        nn.init.xavier_uniform_(new_param)
                    else:
                        nn.init.zeros_(new_param)
                    submod.register_parameter(param_name, new_param)
            for child in submod.children():
                materialize_submodule(child)
        materialize_submodule(module)
        module.to(device)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    print("Loading allenai/reward-bench dataset...")
    from datasets import load_dataset
    dataset = load_dataset("allenai/reward-bench", split="filtered")
    eval_dataset = EvalPreferenceDataset(dataset, tokenizer, max_length=MAX_LENGTH)
    eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Storage for URM diagnostics
    urm_sigmas = []
    urm_expected_rewards = []
    urm_final_vars = []
    urm_incorrect = []
    
    # 1. Load & Evaluate Baseline URM Model
    print("Loading Baseline URM model...")
    baseline_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    baseline_model.eval()
    
    print("Evaluating Baseline URM model over RM-Bench...")
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="URM Eval"):
            chosen_ids = batch["chosen_input_ids"].to(device)
            chosen_mask = batch["chosen_attention_mask"].to(device)
            rejected_ids = batch["rejected_input_ids"].to(device)
            rejected_mask = batch["rejected_attention_mask"].to(device)
            
            # Chosen
            pooled_logits_c, weights_c = baseline_model(input_ids=chosen_ids, attention_mask=chosen_mask, return_dict=False)
            mu_c = pooled_logits_c.view(-1, 5, 2)[:, :, 0]
            sigma_c = F.softplus(pooled_logits_c.view(-1, 5, 2)[:, :, 1]) + 1e-6
            scores_c = (mu_c * weights_c).sum(dim=-1)
            
            # Rejected
            pooled_logits_r, weights_r = baseline_model(input_ids=rejected_ids, attention_mask=rejected_mask, return_dict=False)
            mu_r = pooled_logits_r.view(-1, 5, 2)[:, :, 0]
            sigma_r = F.softplus(pooled_logits_r.view(-1, 5, 2)[:, :, 1]) + 1e-6
            scores_r = (mu_r * weights_r).sum(dim=-1)
            
            urm_sigmas.extend(sigma_c.cpu().float().numpy())
            urm_expected_rewards.extend(scores_c.cpu().float().numpy())
            
            c_var = (weights_c**2 * sigma_c**2).sum(dim=-1)
            r_var = (weights_r**2 * sigma_r**2).sum(dim=-1)
            urm_final_vars.extend((c_var + r_var).cpu().float().numpy())
            
            urm_inc = (scores_c <= scores_r).cpu().float().numpy()
            urm_incorrect.extend(urm_inc)
            
    # Cleanup baseline model
    print("Cleaning up Baseline URM model from memory...")
    del baseline_model
    gc.collect()
    torch.cuda.empty_cache()
    
    # Convert URM arrays
    urm_sigmas = np.array(urm_sigmas)             # [N, 5]
    urm_expected_rewards = np.array(urm_expected_rewards) # [N]
    urm_final_vars = np.array(urm_final_vars)     # [N]
    urm_incorrect = np.array(urm_incorrect)       # [N]

    # Storage for MDN diagnostics
    mdn_mixture_vars = []
    mdn_expected_rewards = []
    mdn_final_vars = []
    mdn_pis = []  # will store shape [N, 5, 3]
    mdn_ss = []   # will store shape [N, 5, 3]
    mdn_mus = []  # will store shape [N, 5, 3]
    mdn_incorrect = []

    # 2. Load & Evaluate MDN-URM Model
    print("Loading MDN-URM model...")
    mdn_model = LlamaForSequenceClassificationWithMDN.from_pretrained(
        MODEL_PATH,
        ignore_mismatched_sizes=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    if mdn_model.score.proj.weight.device.type == "meta" or mdn_model.weights.fc[0].weight.device.type == "meta":
        remove_hooks_and_materialize_meta_parameters(mdn_model, device)
        
    mdn_model.score.load_state_dict(torch.load(MDN_HEAD_WEIGHTS, map_location="cpu"))
    mdn_model.weights.load_state_dict(torch.load(GATING_WEIGHTS, map_location="cpu"))
    mdn_model.eval()

    print("Evaluating MDN-URM model over RM-Bench...")
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="MDN Eval"):
            chosen_ids = batch["chosen_input_ids"].to(device)
            chosen_mask = batch["chosen_attention_mask"].to(device)
            rejected_ids = batch["rejected_input_ids"].to(device)
            rejected_mask = batch["rejected_attention_mask"].to(device)
            
            # Chosen
            scores_mdn_c, weights_mdn_c, mean_mdn_c, params_mdn_c = mdn_model(input_ids=chosen_ids, attention_mask=chosen_mask, return_dict=False)
            # Rejected
            scores_mdn_r, weights_mdn_r, mean_mdn_r, params_mdn_r = mdn_model(input_ids=rejected_ids, attention_mask=rejected_mask, return_dict=False)
            
            pi_c, mu_mdn_c, s_c = params_mdn_c
            pi_r, mu_mdn_r, s_r = params_mdn_r
            
            attr_vars_c = mixture_variance(pi_c, mu_mdn_c, s_c)
            attr_vars_r = mixture_variance(pi_r, mu_mdn_r, s_r)
            
            mdn_mixture_vars.extend(attr_vars_c.cpu().float().numpy())
            mdn_expected_rewards.extend(scores_mdn_c.squeeze(-1).cpu().float().numpy())
            
            mdn_pis.extend(pi_c.cpu().float().numpy())
            mdn_ss.extend(s_c.cpu().float().numpy())
            mdn_mus.extend(mu_mdn_c.cpu().float().numpy())
            
            c_var_mdn = (weights_mdn_c**2 * attr_vars_c).sum(dim=-1)
            r_var_mdn = (weights_mdn_r**2 * attr_vars_r).sum(dim=-1)
            mdn_final_vars.extend((c_var_mdn + r_var_mdn).cpu().float().numpy())
            
            mdn_inc = (scores_mdn_c.squeeze(-1) <= scores_mdn_r.squeeze(-1)).cpu().float().numpy()
            mdn_incorrect.extend(mdn_inc)

    # Cleanup MDN model
    print("Cleaning up MDN model from memory...")
    del mdn_model
    gc.collect()
    torch.cuda.empty_cache()

    # Convert MDN arrays
    mdn_mixture_vars = np.array(mdn_mixture_vars) # [N, 5]
    mdn_expected_rewards = np.array(mdn_expected_rewards) # [N]
    mdn_final_vars = np.array(mdn_final_vars)     # [N]
    mdn_pis = np.array(mdn_pis)                   # [N, 5, 3]
    mdn_ss = np.array(mdn_ss)                     # [N, 5, 3]
    mdn_mus = np.array(mdn_mus)                   # [N, 5, 3]
    mdn_incorrect = np.array(mdn_incorrect)       # [N]

    os.makedirs("results/plots", exist_ok=True)
    attributes = ["Helpfulness", "Correctness", "Coherence", "Complexity", "Verbosity"]
    
    # -------------------------------------------------------------
    # Plot A & B: Histogram of sigma (URM) vs mixture variance (MDN)
    # -------------------------------------------------------------
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(urm_sigmas.flatten(), bins=50, color="orange", alpha=0.7, edgecolor="black")
    plt.title("Standard URM: Sigmas ($\sigma$)")
    plt.xlabel("Sigma Value")
    plt.ylabel("Frequency")
    
    plt.subplot(1, 2, 2)
    plt.hist(mdn_mixture_vars.flatten(), bins=50, color="royalblue", alpha=0.7, edgecolor="black")
    plt.title("MDN-URM: Mixture Variance ($Var(r)$)")
    plt.xlabel("Variance Value")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("results/plots/plot_variance_histograms.png", dpi=300)
    plt.close()
    
    # -------------------------------------------------------------
    # Plot C: Expected Reward Histogram (URM vs MDN)
    # -------------------------------------------------------------
    plt.figure(figsize=(8, 6))
    plt.hist(urm_expected_rewards, bins=50, color="orange", alpha=0.5, label="URM Expected Reward", edgecolor="orange")
    plt.hist(mdn_expected_rewards, bins=50, color="royalblue", alpha=0.5, label="MDN Expected Reward", edgecolor="royalblue")
    plt.title("Expected Reward Distribution on RM-Bench")
    plt.xlabel("Reward Score")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig("results/plots/plot_reward_distribution.png", dpi=300)
    plt.close()
    
    # -------------------------------------------------------------
    # Plot D: Component Usage (pi)
    # -------------------------------------------------------------
    avg_pi = np.mean(mdn_pis, axis=0) # [5, 3]
    x = np.arange(len(attributes))
    width = 0.25
    
    plt.figure(figsize=(10, 6))
    plt.bar(x - width, avg_pi[:, 0], width, label="Component 1", color="cornflowerblue")
    plt.bar(x, avg_pi[:, 1], width, label="Component 2", color="royalblue")
    plt.bar(x + width, avg_pi[:, 2], width, label="Component 3", color="navy")
    plt.xticks(x, attributes)
    plt.ylabel("Mean Mixture Weight ($\pi$)")
    plt.title("Mean Component Weights Per Attribute")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.savefig("results/plots/plot_component_weights.png", dpi=300)
    plt.close()
    
    # -------------------------------------------------------------
    # Plot E: Mean Scales (s)
    # -------------------------------------------------------------
    avg_s = np.mean(mdn_ss, axis=0) # [5, 3]
    plt.figure(figsize=(10, 6))
    plt.bar(x - width, avg_s[:, 0], width, label="Scale 1 ($s_1$)", color="peachpuff")
    plt.bar(x, avg_s[:, 1], width, label="Scale 2 ($s_2$)", color="orange")
    plt.bar(x + width, avg_s[:, 2], width, label="Scale 3 ($s_3$)", color="darkorange")
    plt.xticks(x, attributes)
    plt.ylabel("Mean Scale Parameter ($s$)")
    plt.title("Mean Scale Parameters Per Attribute")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.savefig("results/plots/plot_scale_parameters.png", dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # Plot F: Correlation Plot (Uncertainty vs Incorrect Prediction)
    # -------------------------------------------------------------
    def compute_binned_errors(uncertainties, incorrects, num_bins=10):
        idx = np.argsort(uncertainties)
        sorted_unc = uncertainties[idx]
        sorted_inc = incorrects[idx]
        
        chunks_unc = np.array_split(sorted_unc, num_bins)
        chunks_inc = np.array_split(sorted_inc, num_bins)
        
        bin_means_unc = [np.mean(chunk) for chunk in chunks_unc]
        bin_error_rates = [np.mean(chunk) for chunk in chunks_inc]
        
        return bin_means_unc, bin_error_rates
        
    urm_bins, urm_errors = compute_binned_errors(urm_final_vars, urm_incorrect)
    mdn_bins, mdn_errors = compute_binned_errors(mdn_final_vars, mdn_incorrect)
    
    plt.figure(figsize=(8, 6))
    plt.plot(urm_bins, urm_errors, marker="s", linestyle="-", color="orange", label="URM (Baseline)")
    plt.plot(mdn_bins, mdn_errors, marker="o", linestyle="-", color="royalblue", label="MDN-URM (Ours)")
    plt.xlabel("Binned Pair Uncertainty (Var_chosen + Var_rejected)")
    plt.ylabel("Incorrect Prediction Rate (Error Rate)")
    plt.title("Uncertainty Calibration: Error Rate vs. Binned Uncertainty")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig("results/plots/plot_uncertainty_calibration.png", dpi=300)
    plt.close()
    
    # Save statistics as text report
    with open("results/plots/diagnostics_summary.txt", "w") as f:
        f.write("=== Diagnostic Statistics ===\n")
        f.write(f"URM Mean Sigmas per Attribute:\n")
        for attr, val in zip(attributes, np.mean(urm_sigmas, axis=0)):
            f.write(f"  {attr}: {val:.4f}\n")
        f.write(f"\nMDN Mean Mixture Variances per Attribute:\n")
        for attr, val in zip(attributes, np.mean(mdn_mixture_vars, axis=0)):
            f.write(f"  {attr}: {val:.4f}\n")
        f.write(f"\nMDN Mean Component Weights (pi) per Attribute:\n")
        for i, attr in enumerate(attributes):
            f.write(f"  {attr}: pi1={avg_pi[i, 0]:.4f}, pi2={avg_pi[i, 1]:.4f}, pi3={avg_pi[i, 2]:.4f}\n")
        f.write(f"\nMDN Mean Scale Parameters (s) per Attribute:\n")
        for i, attr in enumerate(attributes):
            f.write(f"  {attr}: s1={avg_s[i, 0]:.4f}, s2={avg_s[i, 1]:.4f}, s3={avg_s[i, 2]:.4f}\n")
        
    print("\n--- Diagnostic Statistics ---")
    print(f"Mean sigmas (URM): {np.mean(urm_sigmas, axis=0)}")
    print(f"Mean mixture vars (MDN): {np.mean(mdn_mixture_vars, axis=0)}")
    print(f"Mean pi weights:\n{avg_pi}")
    print(f"Mean scale params:\n{avg_s}")
    print("Done generating plots!")

if __name__ == "__main__":
    main()

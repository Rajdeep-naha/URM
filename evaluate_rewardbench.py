import argparse
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, LlamaConfig
from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
import csv
from tqdm import tqdm

from models.modeling_mdn_urm import LlamaForSequenceClassificationWithMDN

from models.distribution_statistics import compute_statistics, mixture_variance, mixture_std, mixture_mad0
from evaluation.scoring import (
    get_urm_score,
    get_urm_uncertainty,
    get_urm_std,
    get_mad_uncertainty,
    risk_aware_variance_score,
    risk_aware_variance_only_score,
    risk_aware_mad_score,
)

# Attributes list
ATTRIBUTES = ["helpfulness", "correctness", "coherence", "complexity", "verbosity"]


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

        # Format chosen
        chosen_conv = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": chosen}
        ]
        chosen_text = self.tokenizer.apply_chat_template(chosen_conv, tokenize=False)

        # Format rejected
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


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MDN-URM on RewardBench and generate abstention curves")
    parser.add_argument("--model_name_or_path", type=str, default="LxzGordon/URM-LLaMa-3.1-8B", help="Model checkpoint path or HF model id")
    parser.add_argument("--mdn_head_weights", type=str, default="checkpoints/stage1/best_mdn_head.pt", help="Path to trained MDN head")
    parser.add_argument("--gating_weights", type=str, default="checkpoints/stage2/best_gating_weights.pt", help="Path to trained gating weights")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (usually 1)")
    parser.add_argument("--save_dir", type=str, default="results", help="Directory to save evaluation results")
    parser.add_argument("--dry_run", action="store_true", help="Run a quick CPU dry-run check")
    parser.add_argument("--device_map", type=str, default="auto", help="Device map configuration")
    parser.add_argument("--num_components", type=int, default=3, help="Number of mixture components")
    parser.add_argument("--gaussian", action="store_true", help="Use a single Gaussian head baseline")
    parser.add_argument("--baseline", action="store_true", help="Evaluate the pre-trained standard URM baseline")
    parser.add_argument("--dataset_name", type=str, default="allenai/reward-bench", help="Hugging Face dataset name")
    parser.add_argument("--dataset_split", type=str, default="filtered", help="Dataset split to evaluate")
    parser.add_argument("--gating_variant", type=str, default="gaussian", choices=["gaussian", "residual"], help="Which gating network implementation to use")
    
    # CLI options for plug-in framework
    parser.add_argument("--reward-score", type=str, default="urm", choices=["urm", "risk_variance", "risk_variance_only", "risk_mad"],
                        help="Scoring strategy to use for evaluating expected reward")
    parser.add_argument("--uncertainty", type=str, default="variance", choices=["variance", "std", "mad"],
                        help="Uncertainty metric to use for sorting abstention curves")
    parser.add_argument("--beta-list", type=float, nargs="+", default=[0.0],
                        help="List of beta correction values to evaluate. If multiple are passed, accuracy will be reported for each.")
    return parser.parse_args()


def remove_hooks_and_materialize_meta_parameters(model, device):
    import torch.nn as nn
    from accelerate.hooks import remove_hook_from_module
    print("Resolving meta parameters and removing offload hooks...")
    
    for module_name in ["score", "weights"]:
        if not hasattr(model, module_name):
            continue
        module = getattr(model, module_name)
        
        # Remove any accelerate hooks recursively
        remove_hook_from_module(module, recurse=True)
        
        # Materialize meta parameters recursively
        def materialize_submodule(submod):
            for param_name, param in list(submod.named_parameters(recurse=False)):
                if param.device.type == "meta":
                    new_param = nn.Parameter(torch.empty_like(param, device=device))
                    # Initialize
                    if "weight" in param_name:
                        nn.init.xavier_uniform_(new_param)
                    else:
                        nn.init.zeros_(new_param)
                    submod.register_parameter(param_name, new_param)
                    print(f"Materialized meta parameter: {param_name} in {submod.__class__.__name__} to {device}")
            
            for child in submod.children():
                materialize_submodule(child)
                
        materialize_submodule(module)
        module.to(device)


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Initialize tokenizer
    if args.dry_run:
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-LlamaForCausalLM")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # 2. Initialize Model
    if args.dry_run:
        print("Initializing tiny config LLaMA model for dry run...")
        config = LlamaConfig(
            vocab_size=len(tokenizer),
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            pad_token_id=tokenizer.pad_token_id or 0,
            bos_token_id=tokenizer.bos_token_id or 1,
            eos_token_id=tokenizer.eos_token_id or 2,
            num_components=args.num_components,
            use_gaussian=args.gaussian,
            gating_cls=ResidualGating if args.gating_variant == "residual" else None
        )
        model = LlamaForSequenceClassificationWithMDN(config)
    elif args.baseline:
        print(f"Loading baseline URM model from {args.model_name_or_path}...")
        from transformers import AutoModelForSequenceClassification
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map=args.device_map,
        )
    else:
        print(f"Loading model from {args.model_name_or_path}...")
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
        model = LlamaForSequenceClassificationWithMDN.from_pretrained(
            args.model_name_or_path,
            ignore_mismatched_sizes=True,
            torch_dtype=dtype,
            device_map=args.device_map,
            num_components=args.num_components,
            use_gaussian=args.gaussian,
            gating_cls=None,
            uncertainty_target="residual" if args.gating_variant == "residual" else "label"
        )
        
        # Check device of score.proj.weight or weights.fc
        if model.score.proj.weight.device.type == "meta" or model.weights.fc[0].weight.device.type == "meta":
            print("[WARNING] Custom parameters are on meta device! Materializing to target device...")
            remove_hooks_and_materialize_meta_parameters(model, device)
        
        # Load trained weights
        if os.path.exists(args.mdn_head_weights):
            print(f"Loading trained MDN Head weights from {args.mdn_head_weights}...")
            model.score.load_state_dict(torch.load(args.mdn_head_weights, map_location="cpu"))
        
        target_gating_weights = args.gating_weights
        if os.path.exists(target_gating_weights):
            print(f"Loading trained Gating weights from {target_gating_weights}...")
            model.weights.load_state_dict(torch.load(target_gating_weights, map_location="cpu"))
        else:
            print(f"Warning: Trained Gating weights not found at {target_gating_weights}. Using randomly initialized gating network.")

    if not hasattr(model, "hf_device_map"):
        model.to(device)
    model.eval()

    # 3. Load Dataset
    if args.dry_run:
        print("Generating mock preference data for dry run evaluation...")
        mock_data = [
            {
                "prompt": f"Prompt {i}",
                "chosen": f"Good answer {i}",
                "rejected": f"Bad answer {i}"
            }
            for i in range(20)
        ]
        eval_dataset = EvalPreferenceDataset(mock_data, tokenizer, max_length=args.max_length)
    else:
        print(f"Loading {args.dataset_name} ({args.dataset_split}) dataset from Hugging Face...")
        dataset = load_dataset(args.dataset_name, split=args.dataset_split)
        eval_dataset = EvalPreferenceDataset(dataset, tokenizer, max_length=args.max_length)

    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)

    # 4. Evaluation Loop
    results_by_beta = {beta: [] for beta in args.beta_list}

    print("Evaluating model...")
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            # Chosen
            chosen_input_ids = batch["chosen_input_ids"].to(device)
            chosen_attention_mask = batch["chosen_attention_mask"].to(device)
            
            if args.baseline:
                pooled_logits, chosen_weights = model(input_ids=chosen_input_ids, attention_mask=chosen_attention_mask, return_dict=False)
                chosen_mu = pooled_logits.view(-1, 5, 2)[:, :, 0]
                chosen_log_sigma = pooled_logits.view(-1, 5, 2)[:, :, 1]
                chosen_sigma = F.softplus(chosen_log_sigma) + 1e-6
                chosen_base_scores = (chosen_mu * chosen_weights).sum(dim=-1)
                chosen_seq_vars = ((chosen_weights ** 2) * (chosen_sigma ** 2)).sum(dim=-1)
                chosen_seq_mads = chosen_seq_vars  # Dummy for baseline
            else:
                chosen_outputs = model(input_ids=chosen_input_ids, attention_mask=chosen_attention_mask, return_dict=False)
                chosen_weights = chosen_outputs[1]  # [B, 5]
                
                if args.gaussian:
                    chosen_mu, chosen_sigma = chosen_outputs[3]
                    chosen_base_scores = get_urm_score(chosen_mu, chosen_weights)
                    chosen_seq_vars = get_urm_uncertainty(chosen_sigma ** 2, chosen_weights)
                    chosen_seq_mads = chosen_seq_vars
                else:
                    chosen_pi, chosen_mu, chosen_s = chosen_outputs[3]
                    stats = compute_statistics(chosen_pi, chosen_mu, chosen_s)
                    chosen_base_scores = get_urm_score(chosen_outputs[2], chosen_weights)
                    chosen_seq_vars = get_urm_uncertainty(stats["variance"], chosen_weights)
                    chosen_seq_mads = get_mad_uncertainty(stats["mad0"], chosen_weights)
            
            # Rejected
            rejected_input_ids = batch["rejected_input_ids"].to(device)
            rejected_attention_mask = batch["rejected_attention_mask"].to(device)
            
            if args.baseline:
                pooled_logits, rejected_weights = model(input_ids=rejected_input_ids, attention_mask=rejected_attention_mask, return_dict=False)
                rejected_mu = pooled_logits.view(-1, 5, 2)[:, :, 0]
                rejected_log_sigma = pooled_logits.view(-1, 5, 2)[:, :, 1]
                rejected_sigma = F.softplus(rejected_log_sigma) + 1e-6
                rejected_base_scores = (rejected_mu * rejected_weights).sum(dim=-1)
                rejected_seq_vars = ((rejected_weights ** 2) * (rejected_sigma ** 2)).sum(dim=-1)
                rejected_seq_mads = rejected_seq_vars  # Dummy for baseline
            else:
                rejected_outputs = model(input_ids=rejected_input_ids, attention_mask=rejected_attention_mask, return_dict=False)
                rejected_weights = rejected_outputs[1]  # [B, 5]
                
                if args.gaussian:
                    rejected_mu, rejected_sigma = rejected_outputs[3]
                    rejected_base_scores = get_urm_score(rejected_mu, rejected_weights)
                    rejected_seq_vars = get_urm_uncertainty(rejected_sigma ** 2, rejected_weights)
                    rejected_seq_mads = rejected_seq_vars
                else:
                    rejected_pi, rejected_mu, rejected_s = rejected_outputs[3]
                    stats = compute_statistics(rejected_pi, rejected_mu, rejected_s)
                    rejected_base_scores = get_urm_score(rejected_outputs[2], rejected_weights)
                    rejected_seq_vars = get_urm_uncertainty(stats["variance"], rejected_weights)
                    rejected_seq_mads = get_mad_uncertainty(stats["mad0"], rejected_weights)

            # Loop over beta values and calculate corrections
            for beta in args.beta_list:
                # Chosen corrected
                if args.reward_score == "urm":
                    chosen_final_scores = chosen_base_scores
                    rejected_final_scores = rejected_base_scores
                elif args.reward_score == "risk_variance":
                    chosen_final_scores = risk_aware_variance_score(chosen_base_scores, chosen_seq_vars, beta)
                    rejected_final_scores = risk_aware_variance_score(rejected_base_scores, rejected_seq_vars, beta)
                elif args.reward_score == "risk_variance_only":
                    chosen_final_scores = risk_aware_variance_only_score(chosen_base_scores, chosen_seq_vars, beta)
                    rejected_final_scores = risk_aware_variance_only_score(rejected_base_scores, rejected_seq_vars, beta)
                elif args.reward_score == "risk_mad":
                    chosen_final_scores = risk_aware_mad_score(chosen_base_scores, chosen_seq_mads, beta)
                    rejected_final_scores = risk_aware_mad_score(rejected_base_scores, rejected_seq_mads, beta)
                
                # Active uncertainty selector
                if args.uncertainty == "variance":
                    chosen_uncertainty = chosen_seq_vars
                    rejected_uncertainty = rejected_seq_vars
                elif args.uncertainty == "std":
                    chosen_uncertainty = torch.sqrt(torch.clamp(chosen_seq_vars, min=1e-12))
                    rejected_uncertainty = torch.sqrt(torch.clamp(rejected_seq_vars, min=1e-12))
                elif args.uncertainty == "mad":
                    chosen_uncertainty = chosen_seq_mads
                    rejected_uncertainty = rejected_seq_mads

                # Store predictions
                for idx in range(chosen_base_scores.shape[0]):
                    c_score = chosen_final_scores[idx].item()
                    r_score = rejected_final_scores[idx].item()
                    c_unc = chosen_uncertainty[idx].item()
                    r_unc = rejected_uncertainty[idx].item()
                    
                    correct = 1.0 if c_score > r_score else 0.0
                    pair_uncertainty = c_unc + r_unc

                    results_by_beta[beta].append({
                        "correct": correct,
                        "uncertainty": pair_uncertainty,
                        "chosen_score": c_score,
                        "rejected_score": r_score,
                        "chosen_var": chosen_seq_vars[idx].item(),
                        "rejected_var": rejected_seq_vars[idx].item()
                    })

    # 5. Compute and Save Abstention Results for every Beta
    for beta in args.beta_list:
        sorted_results = sorted(results_by_beta[beta], key=lambda x: x["uncertainty"])
        total_pairs = len(sorted_results)
        
        retain_percentages = [100, 95, 90, 85, 80, 75, 70, 60, 50]
        abstention_results = {}

        print(f"\n--- Uncertainty-Based Abstention Results (Beta = {beta}) ---")
        for pct in retain_percentages:
            num_to_retain = int(total_pairs * (pct / 100.0))
            retained = sorted_results[:num_to_retain]
            
            if num_to_retain > 0:
                accuracy = sum(r["correct"] for r in retained) / num_to_retain
            else:
                accuracy = 0.0
                
            abstention_results[pct] = accuracy
            print(f"Retain {pct}%: Accuracy = {accuracy:.4f} (Retained {num_to_retain}/{total_pairs} pairs)")

        # Save results to CSV
        beta_str = f"_beta_{beta}" if len(args.beta_list) > 1 else ""
        csv_name = f"abstention_results{beta_str}.csv"
        csv_path = os.path.join(args.save_dir, csv_name)
        with open(csv_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Retain Percentage", "Accuracy"])
            for pct, acc in abstention_results.items():
                writer.writerow([f"{pct}%", f"{acc:.6f}"])
        print(f"Abstention results saved to {csv_path}")

        # Generate Plot for single beta run or print done
        if len(args.beta_list) == 1:
            plt.figure(figsize=(8, 6))
            x_val = list(abstention_results.keys())
            y_val = [abstention_results[x] * 100 for x in x_val]
            
            plt.plot(x_val, y_val, marker="o", linewidth=2, color="royalblue")
            plt.xlabel(f"Retain % (Sorted by Uncertainty ({args.uncertainty}) Ascending)")
            plt.ylabel("Accuracy (%)")
            plt.title(f"Uncertainty-Abstention Curve (Beta = {beta}, Score = {args.reward_score})")
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.gca().invert_xaxis()
            
            plot_path = os.path.join(args.save_dir, f"abstention_curve{beta_str}.png")
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            print(f"Abstention plot saved to {plot_path}")


if __name__ == "__main__":
    main()

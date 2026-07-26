import argparse
import os
import gc
import csv
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr, ttest_ind

from models.modeling_mdn_urm import LlamaForSequenceClassificationWithMDN
from models.distribution_statistics import compute_statistics, mixture_variance_decomposition
from evaluation.scoring import (
    get_urm_score,
    get_urm_uncertainty,
    get_urm_std,
    get_mad_uncertainty,
    risk_aware_variance_score,
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
        category = item.get("category", "")
        subcategory = item.get("subcategory", "")

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
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "category": category,
            "subcategory": subcategory
        }


def parse_args():
    parser = argparse.ArgumentParser(description="Run comparison evaluations for MDN Plug-in Uncertainty Framework")
    parser.add_argument("--model_name_or_path", type=str, default="/localstorage/home/f20221218/URM-LLaMa-3.1-8B", help="Base model path")
    parser.add_argument("--label_mdn_head_weights", type=str, default="checkpoints/stage1/best_mdn_head_label.pt", help="Path to trained label MDN head")
    parser.add_argument("--residual_mdn_head_weights", type=str, default="checkpoints/stage1/best_mdn_head_residual.pt", help="Path to trained residual MDN head")
    parser.add_argument("--gating_weights", type=str, default="checkpoints/stage2/best_gating_weights.pt", help="Path to trained gating weights")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--batch_size", type=int, default=4, help="Evaluation batch size")
    parser.add_argument("--save_dir", type=str, default="results", help="Directory to save evaluation results")
    parser.add_argument("--dry_run", action="store_true", help="Run a quick CPU dry-run check")
    parser.add_argument("--device_map", type=str, default="auto", help="Device map configuration")
    parser.add_argument("--num_components", type=int, default=3, help="Number of mixture components")
    parser.add_argument("--dataset_name", type=str, default="allenai/reward-bench", help="Hugging Face dataset name")
    parser.add_argument("--dataset_split", type=str, default="filtered", help="Dataset split to evaluate")
    parser.add_argument("--beta_list", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0, 1.5, 2.0],
                        help="List of beta correction values to evaluate")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples from the dataset to evaluate (useful for quick runs/CPU testing)")
    return parser.parse_args()


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


def evaluate_model_pass(model, eval_loader, device, is_baseline=False):
    """
    Run single-pass model forward evaluation to cache all parameters.
    """
    model.eval()
    
    # We gather everything into lists
    data_dict = {
        "chosen_scores": [], "rejected_scores": [],
        "chosen_sigmas": [], "rejected_sigmas": [],
        "chosen_weights": [], "rejected_weights": [],
        "chosen_pi": [], "chosen_mu": [], "chosen_s": [],
        "rejected_pi": [], "rejected_mu": [], "rejected_s": [],
        "prompts": [], "chosen_texts": [], "rejected_texts": [],
        "categories": [], "subcategories": []
    }
    
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Forward pass"):
            data_dict["prompts"].extend(batch["prompt"])
            data_dict["chosen_texts"].extend(batch["chosen"])
            data_dict["rejected_texts"].extend(batch["rejected"])
            data_dict["categories"].extend(batch["category"])
            data_dict["subcategories"].extend(batch["subcategory"])
            
            c_ids = batch["chosen_input_ids"].to(device)
            c_mask = batch["chosen_attention_mask"].to(device)
            r_ids = batch["rejected_input_ids"].to(device)
            r_mask = batch["rejected_attention_mask"].to(device)
            
            if is_baseline:
                pooled_logits_c, weights_c = model(input_ids=c_ids, attention_mask=c_mask, return_dict=False)
                mu_c = pooled_logits_c.view(-1, 5, 2)[:, :, 0]
                sigma_c = F.softplus(pooled_logits_c.view(-1, 5, 2)[:, :, 1]) + 1e-6
                scores_c = (mu_c * weights_c).sum(dim=-1)
                
                pooled_logits_r, weights_r = model(input_ids=r_ids, attention_mask=r_mask, return_dict=False)
                mu_r = pooled_logits_r.view(-1, 5, 2)[:, :, 0]
                sigma_r = F.softplus(pooled_logits_r.view(-1, 5, 2)[:, :, 1]) + 1e-6
                scores_r = (mu_r * weights_r).sum(dim=-1)
                
                data_dict["chosen_scores"].extend(scores_c.cpu().float())
                data_dict["rejected_scores"].extend(scores_r.cpu().float())
                data_dict["chosen_mu"].extend(mu_c.cpu().float())
                data_dict["rejected_mu"].extend(mu_r.cpu().float())
                data_dict["chosen_sigmas"].extend(sigma_c.cpu().float())
                data_dict["rejected_sigmas"].extend(sigma_r.cpu().float())
                data_dict["chosen_weights"].extend(weights_c.cpu().float())
                data_dict["rejected_weights"].extend(weights_r.cpu().float())
            else:
                # MDN models
                scores_c, w_c, _, params_c = model(input_ids=c_ids, attention_mask=c_mask, return_dict=False)
                scores_r, w_r, _, params_r = model(input_ids=r_ids, attention_mask=r_mask, return_dict=False)
                
                data_dict["chosen_scores"].extend(scores_c.squeeze(-1).cpu().float())
                data_dict["rejected_scores"].extend(scores_r.squeeze(-1).cpu().float())
                data_dict["chosen_weights"].extend(w_c.cpu().float())
                data_dict["rejected_weights"].extend(w_r.cpu().float())
                
                if model.score.use_gaussian:
                    mu_c, sigma_c = params_c
                    mu_r, sigma_r = params_r
                    data_dict["chosen_sigmas"].extend(sigma_c.cpu().float())
                    data_dict["rejected_sigmas"].extend(sigma_r.cpu().float())
                else:
                    pi_c, mu_c, s_c = params_c
                    pi_r, mu_r, s_r = params_r
                    data_dict["chosen_pi"].extend(pi_c.cpu().float())
                    data_dict["chosen_mu"].extend(mu_c.cpu().float())
                    data_dict["chosen_s"].extend(s_c.cpu().float())
                    data_dict["rejected_pi"].extend(pi_r.cpu().float())
                    data_dict["rejected_mu"].extend(mu_r.cpu().float())
                    data_dict["rejected_s"].extend(s_r.cpu().float())
                    
    # Convert lists to tensors where appropriate
    tensor_keys = ["chosen_scores", "rejected_scores", "chosen_sigmas", "rejected_sigmas",
                   "chosen_weights", "rejected_weights", "chosen_pi", "chosen_mu", "chosen_s",
                   "rejected_pi", "rejected_mu", "rejected_s"]
    for k in tensor_keys:
        if len(data_dict[k]) > 0:
            data_dict[k] = torch.stack(data_dict[k]) if isinstance(data_dict[k][0], torch.Tensor) else torch.tensor(data_dict[k])
            
    return data_dict


def process_model_results(cache, beta_list, name="MDN"):
    """
    Process cached predictions to evaluate the sweeps and return list of config results.
    """
    # 1. Expected scores
    ch_base_score = cache["chosen_scores"]
    rj_base_score = cache["rejected_scores"]
    
    # 2. Sequence-level uncertainties
    if "chosen_pi" in cache and len(cache["chosen_pi"]) > 0:
        # MDN model
        ch_pi, ch_mu, ch_s, ch_w = cache["chosen_pi"], cache["chosen_mu"], cache["chosen_s"], cache["chosen_weights"]
        rj_pi, rj_mu, rj_s, rj_w = cache["rejected_pi"], cache["rejected_mu"], cache["rejected_s"], cache["rejected_weights"]
        
        ch_stats = compute_statistics(ch_pi, ch_mu, ch_s)
        rj_stats = compute_statistics(rj_pi, rj_mu, rj_s)
        
        ch_seq_var = get_urm_uncertainty(ch_stats["variance"], ch_w)
        rj_seq_var = get_urm_uncertainty(rj_stats["variance"], rj_w)
        pair_var = (ch_seq_var + rj_seq_var).numpy()
        
        ch_seq_std = get_urm_std(ch_stats["variance"], ch_w)
        rj_seq_std = get_urm_std(rj_stats["variance"], rj_w)
        pair_std = (ch_seq_std + rj_seq_std).numpy()
        
        ch_seq_mad = get_mad_uncertainty(ch_stats["mad0"], ch_w)
        rj_seq_mad = get_mad_uncertainty(rj_stats["mad0"], rj_w)
        pair_mad = (ch_seq_mad + rj_seq_mad).numpy()
        
        # Decompositions
        ch_within, ch_between = mixture_variance_decomposition(ch_pi, ch_mu, ch_s)
        rj_within, rj_between = mixture_variance_decomposition(rj_pi, rj_mu, rj_s)
        
        ch_seq_within = get_urm_uncertainty(ch_within, ch_w)
        rj_seq_within = get_urm_uncertainty(rj_within, rj_w)
        pair_within_var = (ch_seq_within + rj_seq_within).numpy()
        
        ch_seq_between = get_urm_uncertainty(ch_between, ch_w)
        rj_seq_between = get_urm_uncertainty(rj_between, rj_w)
        pair_between_var = (ch_seq_between + rj_seq_between).numpy()
    else:
        # Gaussian URM or Gaussian baseline
        ch_sigma, ch_w = cache["chosen_sigmas"], cache["chosen_weights"]
        rj_sigma, rj_w = cache["rejected_sigmas"], cache["rejected_weights"]
        
        ch_seq_var = get_urm_uncertainty(ch_sigma ** 2, ch_w)
        rj_seq_var = get_urm_uncertainty(rj_sigma ** 2, rj_w)
        pair_var = (ch_seq_var + rj_seq_var).numpy()
        
        ch_seq_std = get_urm_std(ch_sigma ** 2, ch_w)
        rj_seq_std = get_urm_std(rj_sigma ** 2, rj_w)
        pair_std = (ch_seq_std + rj_seq_std).numpy()
        
        import math
        ch_mad0 = ch_sigma * math.sqrt(2.0 / math.pi)
        rj_mad0 = rj_sigma * math.sqrt(2.0 / math.pi)
        ch_seq_mad = get_mad_uncertainty(ch_mad0, ch_w)
        rj_seq_mad = get_mad_uncertainty(rj_mad0, rj_w)
        pair_mad = (ch_seq_mad + rj_seq_mad).numpy()
        
        pair_within_var = pair_var
        pair_between_var = np.zeros_like(pair_var)

    # Base configuration: expected URM score (Beta = 0)
    configs = [{
        "name": f"{name} Mean",
        "chosen_scores": ch_base_score.numpy(),
        "rejected_scores": rj_base_score.numpy(),
        "uncertainty": pair_std,
        "pair_var": pair_var,
        "pair_std": pair_std,
        "pair_mad": pair_mad,
        "pair_within_var": pair_within_var,
        "pair_between_var": pair_between_var
    }]

    for beta in beta_list:
        # SD-corrected
        configs.append({
            "name": f"{name} Mean - {beta}*SD",
            "chosen_scores": risk_aware_variance_score(ch_base_score, ch_seq_var, beta).numpy(),
            "rejected_scores": risk_aware_variance_score(rj_base_score, rj_seq_var, beta).numpy(),
            "uncertainty": pair_std,
            "pair_var": pair_var,
            "pair_std": pair_std,
            "pair_mad": pair_mad,
            "pair_within_var": pair_within_var,
            "pair_between_var": pair_between_var
        })
        # MAD-corrected
        configs.append({
            "name": f"{name} Mean - {beta}*MAD",
            "chosen_scores": risk_aware_mad_score(ch_base_score, ch_seq_mad, beta).numpy(),
            "rejected_scores": risk_aware_mad_score(rj_base_score, rj_seq_mad, beta).numpy(),
            "uncertainty": pair_mad,
            "pair_var": pair_var,
            "pair_std": pair_std,
            "pair_mad": pair_mad,
            "pair_within_var": pair_within_var,
            "pair_between_var": pair_between_var
        })

    # Evaluate each config
    rows = []
    for cfg in configs:
        # Decision Rule: Correct if chosen_score > rejected_score
        correct = (cfg["chosen_scores"] > cfg["rejected_scores"]).astype(float)
        acc = np.mean(correct)
        
        diff = cfg["chosen_scores"] - cfg["rejected_scores"]
        probs = 1.0 / (1.0 + np.exp(-diff))
        nll = -np.mean(np.log(probs + 1e-12))
        
        # Correlation between uncertainty and prediction error
        incorrect = 1.0 - correct
        pearson_r_val, _ = pearsonr(cfg["uncertainty"], incorrect)
        spearman_rho_val, _ = spearmanr(cfg["uncertainty"], incorrect)
        
        # Uncertainty Gap
        unc_correct = cfg["uncertainty"][correct == 1.0]
        unc_incorrect = cfg["uncertainty"][correct == 0.0]
        
        mean_unc_c = np.mean(unc_correct) if len(unc_correct) > 0 else 0.0
        mean_unc_i = np.mean(unc_incorrect) if len(unc_incorrect) > 0 else 0.0
        gap = mean_unc_i - mean_unc_c
        
        if len(unc_correct) > 1 and len(unc_incorrect) > 1:
            t_stat, p_val = ttest_ind(unc_incorrect, unc_correct, equal_var=False)
        else:
            t_stat, p_val = 0.0, 1.0
            
        rows.append({
            "Method": cfg["name"],
            "Accuracy": acc,
            "NLL": nll,
            "Pearson r": pearson_r_val,
            "Spearman rho": spearman_rho_val,
            "Mean Uncertainty (Correct)": mean_unc_c,
            "Mean Uncertainty (Incorrect)": mean_unc_i,
            "Uncertainty Gap": gap,
            "p-value (T-test)": p_val,
            "cfg": cfg
        })
        
    return rows


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, "plots"), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    if args.dry_run:
        print("Generating mock preference data for dry run...")
        mock_data = [
            {
                "prompt": f"Prompt {i}",
                "chosen": f"Chosen response text {i}",
                "rejected": f"Rejected response text {i}",
                "category": "Chat" if i % 2 == 0 else "Safety",
                "subcategory": "General Chat" if i % 2 == 0 else "Harmful Content"
            }
            for i in range(40)
        ]
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-LlamaForCausalLM")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        eval_dataset = EvalPreferenceDataset(mock_data, tokenizer, max_length=args.max_length)
    else:
        print(f"Loading {args.dataset_name} ({args.dataset_split}) dataset from Hugging Face...")
        from datasets import load_dataset
        dataset = load_dataset(args.dataset_name, split=args.dataset_split)
        if args.max_samples is not None:
            print(f"Selecting first {args.max_samples} samples from the dataset...")
            dataset = dataset.select(range(min(args.max_samples, len(dataset))))
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        eval_dataset = EvalPreferenceDataset(dataset, tokenizer, max_length=args.max_length)

    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)

    # -------------------------------------------------------------------------
    # PART 1: Evaluate Baseline URM Model
    # -------------------------------------------------------------------------
    baseline_cache_path = os.path.join(args.save_dir, "baseline_predictions.pt")
    if os.path.exists(baseline_cache_path):
        print(f"Loading cached Baseline URM outputs from {baseline_cache_path}...")
        baseline_cache = torch.load(baseline_cache_path, map_location="cpu")
    else:
        print("\n=== Running Baseline URM Evaluation ===")
        if args.dry_run:
            N = len(eval_dataset)
            baseline_cache = {
                "chosen_scores": torch.randn(N), "rejected_scores": torch.randn(N),
                "chosen_sigmas": torch.rand(N, 5) * 0.1, "rejected_sigmas": torch.rand(N, 5) * 0.1,
                "chosen_weights": F.softmax(torch.randn(N, 5), dim=-1), "rejected_weights": F.softmax(torch.randn(N, 5), dim=-1),
                "prompts": [item["prompt"] for item in mock_data],
                "chosen_texts": [item["chosen"] for item in mock_data],
                "rejected_texts": [item["rejected"] for item in mock_data],
                "categories": [item["category"] for item in mock_data],
                "subcategories": [item["subcategory"] for item in mock_data]
            }
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                args.model_name_or_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map=args.device_map,
            )
            baseline_cache = evaluate_model_pass(model, eval_loader, device, is_baseline=True)
            torch.save(baseline_cache, baseline_cache_path)
            del model
            gc.collect()
            torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # PART 2: Evaluate Label MDN
    # -------------------------------------------------------------------------
    label_cache_path = os.path.join(args.save_dir, "label_predictions.pt")
    if os.path.exists(label_cache_path):
        print(f"Loading cached Label MDN outputs from {label_cache_path}...")
        label_cache = torch.load(label_cache_path, map_location="cpu")
    else:
        if os.path.exists(args.label_mdn_head_weights):
            print("\n=== Running Label MDN-URM Evaluation ===")
            if args.dry_run:
                N = len(eval_dataset)
                label_cache = {
                    "chosen_scores": torch.randn(N), "rejected_scores": torch.randn(N),
                    "chosen_pi": F.softmax(torch.randn(N, 5, 3), dim=-1), "chosen_mu": torch.randn(N, 5, 3), "chosen_s": torch.rand(N, 5, 3) * 0.5 + 0.1,
                    "rejected_pi": F.softmax(torch.randn(N, 5, 3), dim=-1), "rejected_mu": torch.randn(N, 5, 3), "rejected_s": torch.rand(N, 5, 3) * 0.5 + 0.1,
                    "chosen_weights": F.softmax(torch.randn(N, 5), dim=-1), "rejected_weights": F.softmax(torch.randn(N, 5), dim=-1),
                    "prompts": [item["prompt"] for item in mock_data], "chosen_texts": [item["chosen"] for item in mock_data], "rejected_texts": [item["rejected"] for item in mock_data],
                    "categories": [item["category"] for item in mock_data], "subcategories": [item["subcategory"] for item in mock_data]
                }
            else:
                model = LlamaForSequenceClassificationWithMDN.from_pretrained(
                    args.model_name_or_path, ignore_mismatched_sizes=True, torch_dtype=torch.bfloat16,
                    device_map=args.device_map, num_components=args.num_components, uncertainty_target="label"
                )
                if model.score.proj.weight.device.type == "meta" or model.weights.fc[0].weight.device.type == "meta":
                    remove_hooks_and_materialize_meta_parameters(model, device)
                model.score.load_state_dict(torch.load(args.label_mdn_head_weights, map_location="cpu"))
                model.weights.load_state_dict(torch.load(args.gating_weights, map_location="cpu"))
                model.score.to(torch.bfloat16)
                model.weights.to(torch.bfloat16)
                
                label_cache = evaluate_model_pass(model, eval_loader, device, is_baseline=False)
                torch.save(label_cache, label_cache_path)
                del model
                gc.collect()
                torch.cuda.empty_cache()
        else:
            print(f"[WARNING] Label MDN head weights not found at {args.label_mdn_head_weights}. Skipping Label MDN evaluation.")
            label_cache = None

    # -------------------------------------------------------------------------
    # PART 3: Evaluate Residual MDN
    # -------------------------------------------------------------------------
    residual_cache_path = os.path.join(args.save_dir, "residual_predictions.pt")
    if os.path.exists(residual_cache_path):
        print(f"Loading cached Residual MDN outputs from {residual_cache_path}...")
        residual_cache = torch.load(residual_cache_path, map_location="cpu")
    else:
        if os.path.exists(args.residual_mdn_head_weights) or args.dry_run:
            print("\n=== Running Residual MDN-URM Evaluation ===")
            if args.dry_run:
                N = len(eval_dataset)
                residual_cache = {
                    "chosen_scores": torch.randn(N), "rejected_scores": torch.randn(N),
                    "chosen_pi": F.softmax(torch.randn(N, 5, 3), dim=-1), "chosen_mu": torch.randn(N, 5, 3), "chosen_s": torch.rand(N, 5, 3) * 0.5 + 0.1,
                    "rejected_pi": F.softmax(torch.randn(N, 5, 3), dim=-1), "rejected_mu": torch.randn(N, 5, 3), "rejected_s": torch.rand(N, 5, 3) * 0.5 + 0.1,
                    "chosen_weights": F.softmax(torch.randn(N, 5), dim=-1), "rejected_weights": F.softmax(torch.randn(N, 5), dim=-1),
                    "prompts": [item["prompt"] for item in mock_data], "chosen_texts": [item["chosen"] for item in mock_data], "rejected_texts": [item["rejected"] for item in mock_data],
                    "categories": [item["category"] for item in mock_data], "subcategories": [item["subcategory"] for item in mock_data]
                }
            else:
                model = LlamaForSequenceClassificationWithMDN.from_pretrained(
                    args.model_name_or_path, ignore_mismatched_sizes=True, torch_dtype=torch.bfloat16,
                    device_map=args.device_map, num_components=args.num_components, uncertainty_target="residual"
                )
                if model.score.proj.weight.device.type == "meta" or model.weights.fc[0].weight.device.type == "meta":
                    remove_hooks_and_materialize_meta_parameters(model, device)
                model.score.load_state_dict(torch.load(args.residual_mdn_head_weights, map_location="cpu"))
                model.weights.load_state_dict(torch.load(args.gating_weights, map_location="cpu"))
                model.score.to(torch.bfloat16)
                model.weights.to(torch.bfloat16)
                
                residual_cache = evaluate_model_pass(model, eval_loader, device, is_baseline=False)
                torch.save(residual_cache, residual_cache_path)
                del model
                gc.collect()
                torch.cuda.empty_cache()
        else:
            print(f"[WARNING] Residual MDN head weights not found at {args.residual_mdn_head_weights}. Skipping Residual MDN evaluation.")
            residual_cache = None

    # -------------------------------------------------------------------------
    # PART 4: Compile Comparative Results
    # -------------------------------------------------------------------------
    print("\n=== Compiling Comparison Metrics ===")
    comparison_rows = []
    
    # 1. Baseline URM
    baseline_rows = process_model_results(baseline_cache, args.beta_list, name="Gaussian URM")
    comparison_rows.extend(baseline_rows)
    
    # 2. Label MDN
    if label_cache is not None:
        label_rows = process_model_results(label_cache, args.beta_list, name="Label MDN")
        comparison_rows.extend(label_rows)
        
    # 3. Residual MDN
    if residual_cache is not None:
        residual_rows = process_model_results(residual_cache, args.beta_list, name="Residual MDN")
        comparison_rows.extend(residual_rows)

    # Save to comparison_table.csv (omit raw 'cfg' dict)
    comparison_path = os.path.join(args.save_dir, "comparison_table.csv")
    with open(comparison_path, mode="w", newline="") as f:
        fieldnames = [k for k in comparison_rows[0].keys() if k != "cfg"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in comparison_rows:
            filtered_row = {k: v for k, v in row.items() if k != "cfg"}
            writer.writerow(filtered_row)
    print(f"Comparative table saved to {comparison_path}")

    # Expose and print stdout summary
    print("\n--- Summary Table ---")
    for row in comparison_rows:
        if "Mean" in row["Method"] and "-" not in row["Method"]:
            print(f"{row['Method']}: Accuracy = {row['Accuracy']:.4f}, NLL = {row['NLL']:.4f}, Gap = {row['Uncertainty Gap']:.4f} (p={row['p-value (T-test)']:.2e})")
        elif "1.0*SD" in row["Method"] or "1.0*MAD" in row["Method"]:
            print(f"{row['Method']}: Accuracy = {row['Accuracy']:.4f}, NLL = {row['NLL']:.4f}, Gap = {row['Uncertainty Gap']:.4f} (p={row['p-value (T-test)']:.2e})")

    # -------------------------------------------------------------------------
    # PART 5: Save Sweeps to rewardbench_results.csv
    # -------------------------------------------------------------------------
    results_path = os.path.join(args.save_dir, "rewardbench_results.csv")
    with open(results_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Method", "Beta", "Accuracy"])
        for row in comparison_rows:
            name = row["Method"]
            acc = row["Accuracy"]
            if "Mean" in name and "-" not in name:
                writer.writerow([name, "0.0", f"{acc:.6f}"])
            elif "-" in name:
                parts = name.split(" - ")
                beta_part = parts[1].split("*")[0]
                writer.writerow([parts[0], beta_part, f"{acc:.6f}"])
    print(f"Sweeps accuracies saved to {results_path}")

    # -------------------------------------------------------------------------
    # PART 6: Compute Abstention Curves
    # -------------------------------------------------------------------------
    # Compare: Gaussian URM, Label MDN Variance, Label MDN MAD, Residual MDN Variance, Residual MDN MAD
    retain_percentages = [100, 95, 90, 85, 80, 75, 70, 60, 50]
    abstention_matrix = {}
    
    # 1. Gaussian URM
    correct_base = (baseline_cache["chosen_scores"] > baseline_cache["rejected_scores"]).numpy().astype(float)
    ch_sig_base = baseline_cache["chosen_sigmas"]
    ch_w_base = baseline_cache["chosen_weights"]
    rj_sig_base = baseline_cache["rejected_sigmas"]
    rj_w_base = baseline_cache["rejected_weights"]
    
    ch_v_base = ((ch_w_base ** 2) * (ch_sig_base ** 2)).sum(dim=-1)
    rj_v_base = ((rj_w_base ** 2) * (rj_sig_base ** 2)).sum(dim=-1)
    unc_base = (ch_v_base + rj_v_base).numpy()
    abstention_matrix["Gaussian URM"] = (correct_base, unc_base)

    # 2. Label MDN
    if label_cache is not None:
        correct_lbl = (label_cache["chosen_scores"] > label_cache["rejected_scores"]).numpy().astype(float)
        # Compute Variance and MAD
        ch_lbl_stats = compute_statistics(label_cache["chosen_pi"], label_cache["chosen_mu"], label_cache["chosen_s"])
        rj_lbl_stats = compute_statistics(label_cache["rejected_pi"], label_cache["rejected_mu"], label_cache["rejected_s"])
        
        var_lbl = (get_urm_uncertainty(ch_lbl_stats["variance"], label_cache["chosen_weights"]) +
                   get_urm_uncertainty(rj_lbl_stats["variance"], label_cache["rejected_weights"])).numpy()
        mad_lbl = (get_mad_uncertainty(ch_lbl_stats["mad0"], label_cache["chosen_weights"]) +
                   get_mad_uncertainty(rj_lbl_stats["mad0"], label_cache["rejected_weights"])).numpy()
        
        abstention_matrix["Label MDN (Variance)"] = (correct_lbl, var_lbl)
        abstention_matrix["Label MDN (MAD)"] = (correct_lbl, mad_lbl)
        
    # 3. Residual MDN
    if residual_cache is not None:
        correct_res = (residual_cache["chosen_scores"] > residual_cache["rejected_scores"]).numpy().astype(float)
        # Compute Variance and MAD
        ch_res_stats = compute_statistics(residual_cache["chosen_pi"], residual_cache["chosen_mu"], residual_cache["chosen_s"])
        rj_res_stats = compute_statistics(residual_cache["rejected_pi"], residual_cache["rejected_mu"], residual_cache["rejected_s"])
        
        var_res = (get_urm_uncertainty(ch_res_stats["variance"], residual_cache["chosen_weights"]) +
                   get_urm_uncertainty(rj_res_stats["variance"], residual_cache["rejected_weights"])).numpy()
        mad_res = (get_mad_uncertainty(ch_res_stats["mad0"], residual_cache["chosen_weights"]) +
                   get_mad_uncertainty(rj_res_stats["mad0"], residual_cache["rejected_weights"])).numpy()
        
        abstention_matrix["Residual MDN (Variance)"] = (correct_res, var_res)
        abstention_matrix["Residual MDN (MAD)"] = (correct_res, mad_res)

    # Save curves to CSV
    abstention_path = os.path.join(args.save_dir, "abstention_results.csv")
    with open(abstention_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Retain Percentage"] + list(abstention_matrix.keys()))
        
        # Calculate retention curves
        curve_values = {name: [] for name in abstention_matrix.keys()}
        for name, (correct_arr, unc_arr) in abstention_matrix.items():
            idx_sorted = np.argsort(unc_arr)
            correct_sorted = correct_arr[idx_sorted]
            total_len = len(correct_sorted)
            
            for pct in retain_percentages:
                n_retain = int(total_len * (pct / 100.0))
                acc = np.mean(correct_sorted[:n_retain]) if n_retain > 0 else 0.0
                curve_values[name].append(acc)
                
        for idx, pct in enumerate(retain_percentages):
            row = [f"{pct}%"]
            for name in abstention_matrix.keys():
                row.append(f"{curve_values[name][idx]:.6f}")
            writer.writerow(row)
    print(f"Abstention retention results saved to {abstention_path}")

    # -------------------------------------------------------------------------
    # PART 7: Plotting
    # -------------------------------------------------------------------------
    # 1. Sweep Accuracy vs Beta Comparison
    plt.figure(figsize=(10, 6))
    colors = {"Label MDN": "royalblue", "Residual MDN": "green"}
    markers = {"SD": "o", "MAD": "^"}
    
    for prefix in ["Label MDN", "Residual MDN"]:
        if prefix == "Label MDN" and label_cache is None:
            continue
        if prefix == "Residual MDN" and residual_cache is None:
            continue
            
        acc_sd = [next(r["Accuracy"] for r in comparison_rows if r["Method"] == f"{prefix} Mean - {beta}*SD") for beta in args.beta_list]
        acc_mad = [next(r["Accuracy"] for r in comparison_rows if r["Method"] == f"{prefix} Mean - {beta}*MAD") for beta in args.beta_list]
        
        plt.plot(args.beta_list, acc_sd, marker=markers["SD"], label=f"{prefix} - \u03b2*SD", color=colors[prefix])
        plt.plot(args.beta_list, acc_mad, marker=markers["MAD"], label=f"{prefix} - \u03b2*MAD", linestyle="--", color=colors[prefix])
        
    plt.axhline(y=baseline_rows[0]["Accuracy"], color="gray", linestyle="--", label="Gaussian URM (Baseline)")
    plt.xlabel("Correction Strength (\u03b2)")
    plt.ylabel("Accuracy on RewardBench")
    plt.title("RewardBench Accuracy vs. Downstream Correction Strength \u03b2")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    
    beta_plot_path = os.path.join(args.save_dir, "plots/rewardbench_beta_comparison.png")
    plt.savefig(beta_plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Retention Curves
    plt.figure(figsize=(9, 6))
    for name in abstention_matrix.keys():
        plt.plot(retain_percentages, [x * 100 for x in curve_values[name]], marker="o", label=name, linewidth=2)
        
    plt.xlabel("Retention Rate (%)")
    plt.ylabel("Accuracy (%)")
    plt.title("Active Abstention Retention Curves")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.gca().invert_xaxis()
    
    abstention_plot_path = os.path.join(args.save_dir, "plots/rewardbench_abstention_curves.png")
    plt.savefig(abstention_plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    # 3. Scatter Plot: Reward Margin vs. Pair Uncertainty (Sanity Check)
    # Plot for the residual MDN model (or label MDN if residual doesn't exist)
    active_cache = residual_cache if residual_cache is not None else label_cache
    active_prefix = "Residual MDN" if residual_cache is not None else "Label MDN"
    
    if active_cache is not None:
        plt.figure(figsize=(8, 6))
        ch_expected = active_cache["chosen_scores"].numpy()
        rj_expected = active_cache["rejected_scores"].numpy()
        reward_margin = np.abs(ch_expected - rj_expected)
        
        # Calculate standard deviation
        ch_stats = compute_statistics(active_cache["chosen_pi"], active_cache["chosen_mu"], active_cache["chosen_s"])
        rj_stats = compute_statistics(active_cache["rejected_pi"], active_cache["rejected_mu"], active_cache["rejected_s"])
        ch_seq_std = get_urm_std(ch_stats["variance"], active_cache["chosen_weights"]).numpy()
        rj_seq_std = get_urm_std(rj_stats["variance"], active_cache["rejected_weights"]).numpy()
        pair_std = ch_seq_std + rj_seq_std
        
        plt.scatter(reward_margin, pair_std, color="royalblue", alpha=0.6, edgecolors="none")
        plt.xlabel("Absolute Reward Margin |R_chosen - R_rejected|")
        plt.ylabel(f"{active_prefix} Pair Uncertainty (SD)")
        plt.title(f"Reward Margin vs. Pair Uncertainty ({active_prefix})")
        plt.grid(True, linestyle="--", alpha=0.5)
        
        margin_plot_path = os.path.join(args.save_dir, "plots/margin_vs_uncertainty.png")
        plt.savefig(margin_plot_path, dpi=300, bbox_inches="tight")
        plt.close()

    print("All comparison tasks finished successfully!")


if __name__ == "__main__":
    main()

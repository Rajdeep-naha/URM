import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, LlamaConfig, get_cosine_schedule_with_warmup
from datasets import load_dataset
import wandb
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import math

from models.modeling_mdn_urm import LlamaForSequenceClassificationWithMDN
from models.mdn_head import mdn_nll_loss, gaussian_nll_loss
from models.distribution_statistics import mixture_mean, mixture_variance

# HelpSteer2 attributes
ATTRIBUTES = ["helpfulness", "correctness", "coherence", "complexity", "verbosity"]


class HelpSteer2Dataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512, means=None, stds=None):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.means = means
        self.stds = stds

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item["prompt"]
        response = item["response"]

        # Format using chat template: user prompt, assistant response
        conversation = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response}
        ]
        text = self.tokenizer.apply_chat_template(conversation, tokenize=False)

        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        labels = [float(item[attr]) for attr in ATTRIBUTES]
        labels = torch.tensor(labels, dtype=torch.float32)
        if self.means is not None and self.stds is not None:
            labels = (labels - self.means) / self.stds

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": labels
        }


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 1: Attribute Regression with MDN Head")
    parser.add_argument("--model_name_or_path", type=str, default="LxzGordon/URM-LLaMa-3.1-8B", help="Model checkpoint path or HF model id")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size per device")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--grad_accum_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--save_dir", type=str, default="checkpoints/stage1", help="Checkpoint directory")
    parser.add_argument("--dry_run", action="store_true", help="Run a quick CPU dry-run check")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--device_map", type=str, default="auto", help="Device map configuration")
    parser.add_argument("--num_components", type=int, default=3, help="Number of mixture components")
    parser.add_argument("--gaussian", action="store_true", help="Train a single Gaussian head baseline with Gaussian NLL")
    
    # Residual uncertainty learning options
    parser.add_argument("--uncertainty_target", type=str, default="label", choices=["label", "residual"],
                        help="Target target for the MDN head (raw labels or residuals after mean prediction)")
    parser.add_argument("--lambda_nll", type=float, default=1.0, help="Loss weight scale coefficient for NLL term in residual target mode")
    parser.add_argument("--lambda_entropy", type=float, default=0.0, help="Optional entropy regularization weight; keep at 0 until mixture usefulness is verified")
    parser.add_argument("--lambda_repulsive", type=float, default=0.0, help="Optional repulsive mean-loss weight; keep at 0 until collapse remains after ordering/init fixes")
    return parser.parse_args()


def compute_mixture_entropy(pi: torch.Tensor) -> torch.Tensor:
    """Mean mixture entropy across batch and attributes."""
    return -torch.sum(pi * torch.log(pi + 1e-10), dim=-1).mean()


def compute_repulsive_mean_loss(mu: torch.Tensor) -> torch.Tensor:
    """Small repulsive penalty that encourages component means to separate."""
    num_components = mu.shape[-1]
    if num_components <= 1:
        return torch.tensor(0.0, device=mu.device, dtype=mu.dtype)

    diffs = mu.unsqueeze(-1) - mu.unsqueeze(-2)
    log_diffs = torch.log(torch.abs(diffs) + 1e-6)
    mask = torch.triu(torch.ones(num_components, num_components, device=mu.device, dtype=mu.dtype), diagonal=1)
    normalizer = mask.sum() * log_diffs.shape[:-2].numel() + 1e-8
    return -((log_diffs * mask).sum() / normalizer)


def remove_hooks_and_materialize_meta_parameters(model, device):
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

    if args.wandb:
        wandb.init(project="mdn-urm-stage1", name=f"attr-regression-{args.uncertainty_target}")

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, "plots"), exist_ok=True)

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
            uncertainty_target=args.uncertainty_target
        )
        model = LlamaForSequenceClassificationWithMDN(config)
    else:
        print(f"Loading pretrained backbone from {args.model_name_or_path}...")
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
        model = LlamaForSequenceClassificationWithMDN.from_pretrained(
            args.model_name_or_path,
            ignore_mismatched_sizes=True,
            torch_dtype=dtype,
            device_map=args.device_map,
            num_components=args.num_components,
            use_gaussian=args.gaussian,
            uncertainty_target=args.uncertainty_target
        )

    # Materialize meta parameters if score is on meta device
    if model.score.proj.weight.device.type == "meta":
        print("[WARNING] model.score.proj.weight is on meta device! Materializing to target device...")
        remove_hooks_and_materialize_meta_parameters(model, device)
    else:
        print("model.score.proj.weight is on a real device. Skipping materialization.")

    # Ensure all custom parameters are cast to the correct dtype
    model.score.to(dtype)
    if hasattr(model, "weights"):
        model.weights.to(dtype)

    if not hasattr(model, "hf_device_map"):
        model.to(device)

    # 3. Freeze Backbone and Gating layer, Unfreeze MDN Head (includes proj and mean_proj in residual target mode)
    print("Freezing LLaMA backbone and gating layers...")
    for param in model.model.parameters():
        param.requires_grad = False
    for param in model.weights.parameters():
        param.requires_grad = False
    
    print("Unfreezing MDN Head...")
    for param in model.score.parameters():
        param.requires_grad = True

    trainable_params = [n for n, p in model.named_parameters() if p.requires_grad]
    print(f"Trainable parameters: {trainable_params}")

    # 4. Load Dataset
    if args.dry_run:
        print("Generating mock data for dry run...")
        mock_data = [
            {
                "prompt": "Hello, how are you?",
                "response": "I am doing well, thank you!",
                "helpfulness": 4, "correctness": 4, "coherence": 4, "complexity": 1, "verbosity": 2
            },
            {
                "prompt": "Explain gravity.",
                "response": "Gravity is a force that pulls objects together.",
                "helpfulness": 3, "correctness": 3, "coherence": 4, "complexity": 2, "verbosity": 2
            }
        ] * 16  # 32 samples
        train_data = mock_data
        val_data = mock_data
    else:
        print("Loading HelpSteer2 dataset from Hugging Face...")
        dataset = load_dataset("nvidia/HelpSteer2")
        train_data = dataset["train"]
        val_data = dataset["validation"]

    attr_means = []
    attr_stds = []
    print("HelpSteer2 dataset label statistics:")
    for attr in ATTRIBUTES:
        vals = [float(x[attr]) for x in train_data]
        mean = np.mean(vals)
        std = np.std(vals)
        if std < 1e-6:
            std = 1.0
        minimum = np.min(vals)
        maximum = np.max(vals)
        print(f"  {attr}: min={minimum}, max={maximum}, mean={mean:.4f}, std={std:.4f}")
        attr_means.append(mean)
        attr_stds.append(std)
        
    attr_means = torch.tensor(attr_means, dtype=torch.float32)
    attr_stds = torch.tensor(attr_stds, dtype=torch.float32)

    train_dataset = HelpSteer2Dataset(train_data, tokenizer, max_length=args.max_length, means=attr_means, stds=attr_stds)
    val_dataset = HelpSteer2Dataset(val_data, tokenizer, max_length=args.max_length, means=attr_means, stds=attr_stds)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # 5. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.score.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    total_steps = len(train_loader) * args.epochs // args.grad_accum_steps
    warmup_steps = int(total_steps * 0.1)
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # 6. Training Loop
    print("Starting training...")
    best_val_loss = float("inf")
    initial_weight = model.score.proj.weight.clone().detach().cpu()
    is_first_opt_step = True

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for step, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device).to(dtype)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
            expected_rewards = outputs[2]  # [B, 5]

            # In residual mode, expected_rewards is output directly from the mean head mean_proj.
            # In label mode, expected_rewards is expected mixture mean.
            if args.uncertainty_target == "residual":
                # Compute L_mean (MSE on labels)
                loss_mean = nn.functional.mse_loss(expected_rewards, labels)
                
                # Compute residual: r = y - y_hat
                residual = labels - expected_rewards
                
                # Compute L_mdn (NLL of residuals)
                if args.gaussian:
                    mu, sigma = outputs[3]
                    loss_nll = gaussian_nll_loss(residual, mu, sigma)
                else:
                    pi, mu, s = outputs[3]
                    loss_nll = mdn_nll_loss(residual, pi, mu, s)

                    if args.lambda_entropy != 0.0:
                        loss_nll = loss_nll - args.lambda_entropy * compute_mixture_entropy(pi)
                    if args.lambda_repulsive != 0.0:
                        loss_nll = loss_nll + args.lambda_repulsive * compute_repulsive_mean_loss(mu)
                
                loss = loss_mean + args.lambda_nll * loss_nll
            else:
                # Label training mode
                if args.gaussian:
                    mu, sigma = outputs[3]
                    loss = gaussian_nll_loss(labels, mu, sigma)
                else:
                    pi, mu, s = outputs[3]
                    loss = mdn_nll_loss(labels, pi, mu, s)

                    if args.lambda_entropy != 0.0:
                        loss = loss - args.lambda_entropy * compute_mixture_entropy(pi)
                    if args.lambda_repulsive != 0.0:
                        loss = loss + args.lambda_repulsive * compute_repulsive_mean_loss(mu)

            loss = loss / args.grad_accum_steps
            loss.backward()

            # Diagnostic check on first step
            if epoch == 0 and step == 0:
                grad = model.score.proj.weight.grad
                if grad is not None:
                    grad_mean = grad.abs().mean().item()
                    print(f"\n[DIAGNOSTIC] Step 0 gradient flow: grad mean = {grad_mean:.8f}")
                else:
                    print("\n[DIAGNOSTIC] Step 0 gradient flow: grad is None!")

            epoch_loss += loss.item() * args.grad_accum_steps

            if (step + 1) % args.grad_accum_steps == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.score.parameters(), max_norm=1.0)
                
                if is_first_opt_step:
                    print(f"\n[DIAGNOSTIC] weight ID in model: {id(model.score.proj.weight)}")
                    print(f"[DIAGNOSTIC] weight ID in optimizer: {id(optimizer.param_groups[0]['params'][0])}")
                    print(f"[DIAGNOSTIC] Current learning rate: {optimizer.param_groups[0]['lr']}")
                    print(f"[DIAGNOSTIC] Weight mean abs before step: {model.score.proj.weight.data.abs().mean().item():.8f}")
                    
                optimizer.step()
                
                if is_first_opt_step:
                    print(f"[DIAGNOSTIC] Weight mean abs after step: {model.score.proj.weight.data.abs().mean().item():.8f}")
                scheduler.step()
                
                if is_first_opt_step:
                    current_weight = model.score.proj.weight.clone().detach().cpu()
                    weight_update = (current_weight - initial_weight).abs().mean().item()
                    print(f"\n[DIAGNOSTIC] First optimizer step weight update: {weight_update:.8f}\n")
                    is_first_opt_step = False

                optimizer.zero_grad()

            progress_bar.set_postfix({"loss": loss.item() * args.grad_accum_steps})

            if args.wandb:
                wandb.log({
                    "train_step_loss": loss.item() * args.grad_accum_steps,
                    "lr": optimizer.param_groups[0]["lr"]
                })

        avg_train_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch + 1} average training loss: {avg_train_loss:.4f}")
        with torch.no_grad():
            weight_magnitude = model.score.proj.weight.abs().mean().item()
            print(f"Epoch {epoch + 1} Head Weight Magnitude: {weight_magnitude:.8f}")

        # -------------------------------------------------------------------------
        # Validation Loop
        # -------------------------------------------------------------------------
        model.eval()
        val_loss = 0.0
        val_rmse = []

        all_expected_rewards_norm = []
        all_expected_rewards_orig = []
        all_labels_norm = []
        all_labels_orig = []
        all_pi = []
        all_mu = []
        all_s = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device).to(dtype)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
                expected_rewards = outputs[2]  # [B, 5]

                # Compute loss corresponding to target target mode
                if args.uncertainty_target == "residual":
                    loss_mean = nn.functional.mse_loss(expected_rewards, labels)
                    residual = labels - expected_rewards
                    if args.gaussian:
                        mu, sigma = outputs[3]
                        loss_nll = gaussian_nll_loss(residual, mu, sigma)
                    else:
                        pi, mu, s = outputs[3]
                        loss_nll = mdn_nll_loss(residual, pi, mu, s)
                    loss = loss_mean + args.lambda_nll * loss_nll
                else:
                    if args.gaussian:
                        mu, sigma = outputs[3]
                        loss = gaussian_nll_loss(labels, mu, sigma)
                    else:
                        pi, mu, s = outputs[3]
                        loss = mdn_nll_loss(labels, pi, mu, s)

                val_loss += loss.item()

                expected_rewards_cpu = expected_rewards.cpu().to(torch.float32)
                expected_rewards_orig = expected_rewards_cpu * attr_stds + attr_means
                labels_norm = labels.cpu().to(torch.float32)
                labels_orig = labels_norm * attr_stds + attr_means

                all_expected_rewards_norm.append(expected_rewards_cpu)
                all_expected_rewards_orig.append(expected_rewards_orig)
                all_labels_norm.append(labels_norm)
                all_labels_orig.append(labels_orig)
                
                if args.gaussian:
                    all_mu.append(mu.cpu())
                    all_s.append(sigma.cpu())
                    all_pi.append(torch.ones_like(mu).cpu())
                else:
                    all_pi.append(pi.cpu())
                    all_mu.append(mu.cpu())
                    all_s.append(s.cpu())

                rmse = torch.sqrt(torch.mean((expected_rewards_orig - labels_orig) ** 2, dim=0))
                val_rmse.append(rmse.numpy())

        avg_val_loss = val_loss / len(val_loader)
        avg_val_rmse = np.mean(val_rmse, axis=0)
        
        all_expected_rewards_norm = torch.cat(all_expected_rewards_norm, dim=0)
        all_expected_rewards_orig = torch.cat(all_expected_rewards_orig, dim=0)
        all_labels_norm = torch.cat(all_labels_norm, dim=0)
        all_labels_orig = torch.cat(all_labels_orig, dim=0)
        all_pi = torch.cat(all_pi, dim=0)
        all_mu = torch.cat(all_mu, dim=0)
        all_s = torch.cat(all_s, dim=0)
        
        residuals_norm = all_labels_norm - all_expected_rewards_norm
        residuals_orig = all_labels_orig - all_expected_rewards_orig
        
        print(f"Epoch {epoch + 1} validation joint loss: {avg_val_loss:.4f}")
        for i, attr in enumerate(ATTRIBUTES):
            print(f"Validation RMSE for {attr}: {avg_val_rmse[i]:.4f}")

        print("\n=== Validation Diagnostics ===")
        print(f"  Target Means (Orig):   {all_labels_orig.mean(dim=0).tolist()}")
        print(f"  Expected Reward Means (Orig): {all_expected_rewards_orig.mean(dim=0).tolist()}")
        print(f"  Mean Residuals (Orig):  {residuals_orig.mean(dim=0).tolist()}")
        print(f"  Residual RMSE (Orig):  {torch.sqrt((residuals_orig ** 2).mean(dim=0)).tolist()}")
        
        pi_mean = all_pi.mean(dim=0)
        pi_entropy = -torch.sum(all_pi * torch.log(all_pi + 1e-10), dim=-1).mean(dim=0)
        mu_mean = all_mu.mean(dim=0)
        mu_std = all_mu.std(dim=0)
        s_mean = all_s.mean(dim=0)
        s_std = all_s.std(dim=0)
        
        for i, attr in enumerate(ATTRIBUTES):
            print(f"  Attribute: {attr}")
            print(f"    pi mean:    {pi_mean[i].tolist()}")
            if not args.gaussian:
                print(f"    pi entropy: {pi_entropy[i].item():.4f}")
            print(f"    mu mean:    {mu_mean[i].tolist()}")
            print(f"    mu std:     {mu_std[i].tolist()}")
            print(f"    s/sigma mean: {s_mean[i].tolist()}")
            print(f"    s/sigma std:  {s_std[i].tolist()}")
        print("==============================\n")

        if args.wandb:
            wandb_log_dict = {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_joint_loss": avg_val_loss,
                "weight_magnitude": weight_magnitude,
            }
            for i, attr in enumerate(ATTRIBUTES):
                wandb_log_dict[f"val_rmse_{attr}"] = avg_val_rmse[i]
                wandb_log_dict[f"mean_residual_{attr}"] = residuals_orig.mean(dim=0)[i].item()
                if not args.gaussian:
                    wandb_log_dict[f"val_pi_entropy_{attr}"] = pi_entropy[i].item()
            wandb.log(wandb_log_dict)

        # Save best model checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = os.path.join(args.save_dir, "best_mdn_head.pt")
            print(f"Saving best model checkpoint to {checkpoint_path}")
            torch.save(model.score.state_dict(), checkpoint_path)
            diagnostics_path = os.path.join(args.save_dir, "best_validation_diagnostics.pt")
            torch.save(
                {
                    "attributes": ATTRIBUTES,
                    "uncertainty_target": args.uncertainty_target,
                    "gaussian": args.gaussian,
                    "labels_norm": all_labels_norm,
                    "labels_orig": all_labels_orig,
                    "expected_rewards_norm": all_expected_rewards_norm,
                    "expected_rewards_orig": all_expected_rewards_orig,
                    "residuals_norm": residuals_norm,
                    "residuals_orig": residuals_orig,
                    "pi": all_pi,
                    "mu": all_mu,
                    "s": all_s,
                    "attr_means": attr_means,
                    "attr_stds": attr_stds,
                },
                diagnostics_path,
            )
            print(f"Saved validation diagnostics to {diagnostics_path}")

    # -------------------------------------------------------------------------
    # Generate and Save Diagnostic Plots at the End of Training
    # -------------------------------------------------------------------------
    print("\nGenerating final diagnostic plots...")
    residuals_orig_np = residuals_orig.numpy()
    
    # Compute predicted uncertainty variance and standard deviation in original scale
    if args.gaussian:
        var_norm = all_s ** 2
    else:
        var_norm = mixture_variance(all_pi, all_mu, all_s)
    
    var_orig = (var_norm * (attr_stds ** 2)).numpy()
    sd_orig = np.sqrt(np.clip(var_orig, 1e-12, None))

    # 1. Residual Histogram
    plt.figure(figsize=(10, 6))
    for i, attr in enumerate(ATTRIBUTES):
        plt.hist(residuals_orig_np[:, i], bins=50, alpha=0.5, label=f"{attr} (mean={residuals_orig_np[:, i].mean():.3f})")
    plt.xlabel("Prediction Error (Residuals)")
    plt.ylabel("Frequency")
    plt.title(f"Residual Error Distributions ({args.uncertainty_target} mode)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    hist_plot_path = os.path.join(args.save_dir, "plots/residual_histogram.png")
    plt.savefig(hist_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved residual histogram to {hist_plot_path}")

    # 2. Predicted Variance vs actual residual squared (Binned)
    # 3. Predicted SD vs absolute residual (Binned)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    num_bins = 10
    for i, attr in enumerate(ATTRIBUTES):
        pred_var = var_orig[:, i]
        squared_err = residuals_orig_np[:, i] ** 2
        
        # Sort and bin by predicted variance
        idx_sorted_var = np.argsort(pred_var)
        pred_var_sorted = pred_var[idx_sorted_var]
        squared_err_sorted = squared_err[idx_sorted_var]
        
        bin_edges = np.linspace(0, len(pred_var), num_bins + 1, dtype=int)
        binned_var = []
        binned_err = []
        
        for b in range(num_bins):
            start, end = bin_edges[b], bin_edges[b+1]
            binned_var.append(pred_var_sorted[start:end].mean())
            binned_err.append(squared_err_sorted[start:end].mean())
            
        ax1.plot(binned_var, binned_err, marker="o", label=attr)
        
        # SD vs Absolute Error
        pred_sd = sd_orig[:, i]
        abs_err = np.abs(residuals_orig_np[:, i])
        idx_sorted_sd = np.argsort(pred_sd)
        pred_sd_sorted = pred_sd[idx_sorted_sd]
        abs_err_sorted = abs_err[idx_sorted_sd]
        
        binned_sd = []
        binned_abs_err = []
        for b in range(num_bins):
            start, end = bin_edges[b], bin_edges[b+1]
            binned_sd.append(pred_sd_sorted[start:end].mean())
            binned_abs_err.append(abs_err_sorted[start:end].mean())
            
        ax2.plot(binned_sd, binned_abs_err, marker="s", label=attr)
        
    ax1.set_xlabel("Mean Predicted Variance")
    ax1.set_ylabel("Mean Actual Squared Error (r^2)")
    ax1.set_title("Predicted Variance vs Empirical Squared Error")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.5)
    
    # Reference line on Variance plot
    all_var_vals = np.concatenate([var_orig[:, i] for i in range(5)])
    ref_x = np.linspace(0, all_var_vals.max(), 100)
    ax1.plot(ref_x, ref_x, color="gray", linestyle="--", alpha=0.7, label="Perfect calibration")
    
    ax2.set_xlabel("Mean Predicted SD")
    ax2.set_ylabel("Mean Actual Absolute Error (|r|)")
    ax2.set_title("Predicted Standard Deviation vs Empirical Absolute Error")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.5)
    
    fig_path = os.path.join(args.save_dir, "plots/predicted_vs_actual_error.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved variance and SD calibration plots to {fig_path}")

    # 4. Continuous Calibration Q-Q PIT Plot
    # Calculate probability integral transform u_i = F(r_i)
    plt.figure(figsize=(8, 8))
    
    # We compute the PIT value using normalized labels and parameters
    for i, attr in enumerate(ATTRIBUTES):
        r_vals = residuals_norm.numpy()[:, i]
        
        pi_attr = all_pi[:, i, :].numpy()
        mu_attr = all_mu[:, i, :].numpy()
        s_attr = all_s[:, i, :].numpy()
        
        pit_values = []
        for j in range(len(r_vals)):
            r = r_vals[j]
            pi = pi_attr[j]
            mu = mu_attr[j]
            s = s_attr[j]
            
            if args.gaussian:
                # Gaussian CDF
                z = (r - mu[0]) / s[0]
                u = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
            else:
                # Logistic Mixture CDF
                # CDF of Logistic is sigmoid((r - mu) / s)
                z = (r - mu) / s
                sig = 1.0 / (1.0 + np.exp(-z))
                u = np.sum(pi * sig)
            pit_values.append(u)
            
        pit_values = np.array(pit_values)
        pit_values_sorted = np.sort(pit_values)
        empirical_cdf = np.arange(1, len(pit_values) + 1) / len(pit_values)
        
        plt.plot(pit_values_sorted, empirical_cdf, label=attr)
        
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Perfect calibration")
    plt.xlabel("Predicted CDF Value (PIT)")
    plt.ylabel("Empirical Cumulative Fraction")
    plt.title(f"PIT Calibration Q-Q Plot ({args.uncertainty_target} mode)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    
    qq_plot_path = os.path.join(args.save_dir, "plots/calibration_qq.png")
    plt.savefig(qq_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved PIT Calibration Q-Q plot to {qq_plot_path}")

    print("Stage 1 Training Complete!")


if __name__ == "__main__":
    main()

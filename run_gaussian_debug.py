import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.stats

ATTRIBUTES = ["helpfulness", "correctness", "coherence", "complexity", "verbosity"]
MODEL_PATH = "/localstorage/home/f20221218/URM-LLaMa-3.1-8B"
SAVE_DIR = "results/gaussian_debug"

os.makedirs(SAVE_DIR, exist_ok=True)

class HelpSteer2Dataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        conversation = [
            {"role": "user", "content": item["prompt"]},
            {"role": "assistant", "content": item["response"]}
        ]
        text = self.tokenizer.apply_chat_template(conversation, tokenize=False)
        inputs = self.tokenizer(text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
        labels = torch.tensor([float(item[attr]) for attr in ATTRIBUTES], dtype=torch.float32)
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": labels,
            "id": idx
        }

def main():
    print(f"--- Step 2: Verify Checkpoint Consistency ---")
    print(f"Loading checkpoint from: {MODEL_PATH}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    
    with open(os.path.join(SAVE_DIR, "checkpoint_report.txt"), "w") as f:
        f.write(f"Checkpoint path: {MODEL_PATH}\n")
        f.write(f"Model class: {model.__class__.__name__}\n")
        f.write(f"Problem type: {model.config.problem_type}\n")
        
    dataset = load_dataset("nvidia/HelpSteer2")
    train_data = dataset["train"]
    val_data = dataset["validation"]
    
    val_dataset = HelpSteer2Dataset(val_data, tokenizer)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    all_labels, all_preds, all_uncs = [], [], []
    
    print("Running inference on validation set...")
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]
            
            # Replicating original evaluation
            batch_size = input_ids.shape[0]
            sequence_lengths = torch.eq(input_ids, model.config.pad_token_id).int().argmax(-1) - 1
            sequence_lengths = sequence_lengths % input_ids.shape[-1]
            sequence_lengths = sequence_lengths.to(device)
            
            transformer_outputs = model.model(input_ids, attention_mask=attention_mask)
            hidden_states = transformer_outputs[0]
            logits = model.score(hidden_states)
            pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]
            
            # Step 4: Verify Output Tensor
            # Original code: rews=pooled_logits.view(-1,5,2)[:,:,0].view(-1,5)
            params = pooled_logits.view(-1, 5, 2).cpu().to(torch.float32)
            mu = params[:, :, 0]
            sigma = torch.exp(params[:, :, 1]) # original URM uses exp for sigma
            
            all_labels.append(labels)
            all_preds.append(mu)
            all_uncs.append(sigma)
            
    labels_val = torch.cat(all_labels, dim=0).numpy()
    preds_val = torch.cat(all_preds, dim=0).numpy()
    uncs_val = torch.cat(all_uncs, dim=0).numpy()
    
    print("--- Step 5: Sample-Level Inspection ---")
    np.random.seed(42)
    sample_indices = np.random.choice(len(labels_val), 20, replace=False)
    
    sample_records = []
    for idx in sample_indices:
        gt = labels_val[idx]
        pred = preds_val[idx]
        unc = uncs_val[idx]
        err = np.abs(gt - pred)
        sample_records.append({
            "Sample ID": idx,
            "Ground Truth": list(np.round(gt, 3)),
            "Predicted mu": list(np.round(pred, 3)),
            "Predicted sigma": list(np.round(unc, 3)),
            "Absolute Error": list(np.round(err, 3))
        })
        print(f"Sample {idx}\nGT\n{np.round(gt, 3).tolist()}\nPrediction\n{np.round(pred, 3).tolist()}\nSigma\n{np.round(unc, 3).tolist()}\n")
        
    pd.DataFrame(sample_records).to_csv(os.path.join(SAVE_DIR, "sample_predictions.csv"), index=False)
    
    print("--- Step 6: Attribute Statistics ---")
    stats_records = []
    for i, attr in enumerate(ATTRIBUTES):
        gt_attr = labels_val[:, i]
        pred_attr = preds_val[:, i]
        
        stats_records.append({
            "Attribute": attr,
            "GT Mean": np.mean(gt_attr),
            "GT Std": np.std(gt_attr),
            "GT Min": np.min(gt_attr),
            "GT Max": np.max(gt_attr),
            "Pred Mean": np.mean(pred_attr),
            "Pred Std": np.std(pred_attr),
            "Pred Min": np.min(pred_attr),
            "Pred Max": np.max(pred_attr),
            "Bias": np.mean(pred_attr - gt_attr)
        })
    pd.DataFrame(stats_records).to_csv(os.path.join(SAVE_DIR, "attribute_statistics.csv"), index=False)
    
    print("--- Step 7: Scatter Plots ---")
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    for i, attr in enumerate(ATTRIBUTES):
        ax = axes[i]
        ax.scatter(labels_val[:, i], preds_val[:, i], alpha=0.1)
        
        min_val = min(np.min(labels_val[:, i]), np.min(preds_val[:, i]))
        max_val = max(np.max(labels_val[:, i]), np.max(preds_val[:, i]))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        ax.set_title(attr)
        ax.set_xlabel("Ground Truth")
        ax.set_ylabel("Predicted mu")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "prediction_vs_groundtruth.png"))
    plt.close()
    
    print("--- Step 8: Residual Distribution ---")
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    for i, attr in enumerate(ATTRIBUTES):
        ax = axes[i]
        res = labels_val[:, i] - preds_val[:, i]
        ax.hist(res, bins=30, alpha=0.7)
        ax.set_title(f"{attr}\nMean={np.mean(res):.3f}, Skew={scipy.stats.skew(res):.3f}")
        ax.set_xlabel("Residual (GT - Pred)")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "residual_histograms.png"))
    plt.close()
    
    print("--- Step 9: Calibration ---")
    cov_records = []
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    for i, attr in enumerate(ATTRIBUTES):
        ax = axes[i]
        gt = labels_val[:, i]
        mu = preds_val[:, i]
        sigma = uncs_val[:, i]
        
        err = np.abs(gt - mu)
        rmse = np.sqrt(np.mean((gt - mu)**2))
        
        var = sigma ** 2
        nll = 0.5 * np.log(2 * np.pi * var) + ((gt - mu) ** 2) / (2 * var)
        mean_nll = np.mean(nll)
        
        in_1s = np.mean(err <= sigma) * 100
        in_2s = np.mean(err <= 2 * sigma) * 100
        in_3s = np.mean(err <= 3 * sigma) * 100
        
        cov_records.append({
            "Attribute": attr,
            "RMSE": rmse,
            "Mean NLL": mean_nll,
            "In 1σ (Exp 68%)": in_1s,
            "In 2σ (Exp 95%)": in_2s,
            "In 3σ (Exp 99.7%)": in_3s
        })
        
        # Reliability plot (binned sigma vs binned RMSE)
        sorted_idx = np.argsort(sigma)
        n_bins = 10
        bin_size = len(sorted_idx) // n_bins
        bin_s, bin_r = [], []
        for b in range(n_bins):
            start = b * bin_size
            end = start + bin_size if b < n_bins - 1 else len(sorted_idx)
            idx = sorted_idx[start:end]
            bin_s.append(np.mean(sigma[idx]))
            bin_r.append(np.sqrt(np.mean(err[idx]**2)))
            
        ax.plot(bin_s, bin_r, 'o-')
        max_s = max(bin_s)
        ax.plot([0, max_s], [0, max_s], 'r--')
        ax.set_title(attr)
        ax.set_xlabel("Predicted Sigma")
        ax.set_ylabel("Actual RMSE")
        
    pd.DataFrame(cov_records).to_csv(os.path.join(SAVE_DIR, "coverage_statistics.csv"), index=False)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "gaussian_calibration.png"))
    plt.close()
    
    print("--- Step 10: Check for Distribution Shift ---")
    print("Computing training stats...")
    train_labels = []
    for i in range(min(10000, len(train_data))): # limit to 10k for speed
        train_labels.append([float(train_data[i][attr]) for attr in ATTRIBUTES])
    train_labels = np.array(train_labels)
    
    for i, attr in enumerate(ATTRIBUTES):
        t_mean = np.mean(train_labels[:, i])
        t_var = np.var(train_labels[:, i])
        v_mean = np.mean(labels_val[:, i])
        v_var = np.var(labels_val[:, i])
        print(f"{attr}: Train Mean={t_mean:.3f}, Var={t_var:.3f} | Val Mean={v_mean:.3f}, Var={v_var:.3f}")

if __name__ == '__main__':
    main()

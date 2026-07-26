"""
Inspect 20 validation samples from HelpSteer2 using the trained Residual MDN model.
Prints: label, prediction, residual, mixture_mean, and predicted_sd to diagnose scaling/logging issues.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

from models.modeling_mdn_urm import LlamaForSequenceClassificationWithMDN
from train_attribute_mdn import HelpSteer2Dataset, ATTRIBUTES
from models.distribution_statistics import compute_statistics

# Setup model
device = "cpu"
tokenizer = AutoTokenizer.from_pretrained("LxzGordon/URM-LLaMa-3.1-8B")
model = LlamaForSequenceClassificationWithMDN.from_pretrained(
    "LxzGordon/URM-LLaMa-3.1-8B",
    trust_remote_code=True,
    torch_dtype=torch.float32,
    device_map=device
)

# Load Stage 1 MDN Head checkpoint
checkpoint_path = "checkpoints/stage1_residual/best_mdn_head.pt"
print(f"Loading checkpoint from {checkpoint_path}...")
model.score.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

# Load HelpSteer2 validation dataset
print("Loading HelpSteer2 dataset...")
dataset = load_dataset("nvidia/HelpSteer2", split="validation")

# Compute dataset statistics (means/stds) just like in training script
raw_labels = []
for item in dataset:
    raw_labels.append([item[attr] for attr in ATTRIBUTES])
raw_labels = np.array(raw_labels)
attr_means = torch.tensor(raw_labels.mean(axis=0), dtype=torch.float32)
attr_stds = torch.tensor(raw_labels.std(axis=0), dtype=torch.float32)

print(f"Dataset Attribute Means: {attr_means.tolist()}")
print(f"Dataset Attribute Stds: {attr_stds.tolist()}")

# Create dataset loader
val_dataset = HelpSteer2Dataset(dataset, tokenizer, max_length=512, means=attr_means, stds=attr_stds)
val_loader = DataLoader(val_dataset, batch_size=20, shuffle=False)

# Get one batch
batch = next(iter(val_loader))
input_ids = batch["input_ids"]
attention_mask = batch["attention_mask"]
labels = batch["labels"] # Normalized targets

with torch.no_grad():
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
    expected_rewards = outputs[2] # Predicted expected rewards (normalized)
    pi, mu, s = outputs[3] # Predicted mixture parameters (residual distribution)

# Select first attribute (helpfulness)
attr_idx = 0
attr_name = ATTRIBUTES[attr_idx]
mean_val = attr_means[attr_idx].item()
std_val = attr_stds[attr_idx].item()

print(f"\n=== Diagnostics for first 20 samples (Attribute: {attr_name}) ===")
print(f"{'Idx':<4} | {'Label (Orig)':<12} | {'Pred (Orig)':<12} | {'Residual (Orig)':<15} | {'Mix Mean (Orig)':<15} | {'Pred SD (Orig)':<14}")
print("-" * 85)

for idx in range(20):
    lbl_norm = labels[idx, attr_idx].item()
    pred_norm = expected_rewards[idx, attr_idx].item()
    
    # Unnormalize to original scale
    lbl_orig = lbl_norm * std_val + mean_val
    pred_orig = pred_norm * std_val + mean_val
    res_orig = lbl_orig - pred_orig
    
    # Mixture stats (residual distribution is on the normalized scale, so we compute SD and scale it)
    pi_sample = pi[idx, :, attr_idx]
    mu_sample = mu[idx, :, attr_idx]
    s_sample = s[idx, :, attr_idx]
    
    # Compute mixture mean and variance in normalized scale
    mix_mean_norm = torch.sum(pi_sample * mu_sample).item()
    logistic_var = (math.pi ** 2 / 3.0) * (s_sample ** 2)
    within_var = torch.sum(pi_sample * logistic_var)
    between_var = torch.sum(pi_sample * ((mu_sample - mix_mean_norm) ** 2))
    total_var_norm = within_var + between_var
    sd_norm = torch.sqrt(total_var_norm).item()
    
    # Scale back mix_mean and sd to original scale
    mix_mean_orig = mix_mean_norm * std_val
    sd_orig = sd_norm * std_val
    
    print(f"{idx:<4} | {lbl_orig:<12.4f} | {pred_orig:<12.4f} | {res_orig:<15.4f} | {mix_mean_orig:<15.4f} | {sd_orig:<14.4f}")

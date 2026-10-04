# URM-MDN: Uncertainty-aware Reward Modelling with Mixture Density Networks

This repository extends the [URM (Uncertainty-aware Reward Model)](https://huggingface.co/LxzGorworworworworworworking/URM-LLaMa-3.1-8B) baseline by replacing its single-Gaussian attribute head with a **Mixture Density Network (MDN)** head, enabling richer uncertainty estimation for reward modelling in RLHF pipelines.

## Overview

```
Frozen LLaMA-3.1-8B backbone
         │
         ├─── MDN Attribute Head (K=3 logistic components × 5 attributes)
         │         │
         │         ├── Label MDN   → models p(label | x)
         │         └── Residual MDN → models p(label − ŷ | x)  ← better for abstention
         │
         └─── Gating Network (5 learned attribute weights)
                   │
                   └── Final Reward R = Σ wⱼ · E[rⱼ]
```

### Key Results (RewardBench)

| Model | Accuracy | Uncertainty-Error Corr. | Abstention Peak |
|---|---|---|---|
| Gaussian URM (baseline) | 87.67% | +0.21 | 93.6% @ 60% retain |
| Residual MDN | 88.41% | **+0.29** | 92.6% @ 85% retain |
| Label MDN | 89.11% | −0.13 (wrong direction) | ❌ degrades |

---

## Repository Structure

```
URM/
├── models/
│   ├── mdn_head.py                 # MDN head: K=3 logistic mixture per attribute
│   ├── modeling_mdn_urm.py         # Full model: backbone + MDN head + gating
│   └── distribution_statistics.py  # Analytical mixture stats (mean, variance, MAD, entropy)
│
├── evaluation/
│   └── scoring.py                  # Scoring functions: URM score, uncertainty, risk-aware scores
│
├── train_attribute_mdn.py          # Stage 1: Train MDN head on HelpSteer2 (label or residual)
├── train_gating.py                 # Stage 2: Train gating network on Skywork-Reward-80K
├── evaluate_rewardbench.py         # Single-model RewardBench evaluation
├── generate_comparison.py          # Multi-model comparison: accuracy, NLL, abstention curves
├── generate_plots.py               # Diagnostic plots (mixture distributions, calibration)
│
├── slurm/                          # SLURM job scripts for cluster
│   ├── train_attribute_mdn.slurm
│   ├── run_residual_train.slurm
│   ├── train_gating.slurm
│   ├── evaluate_rewardbench.slurm
│   ├── run_all_rewardbench.slurm
│   ├── run_plugin_eval.slurm
│   └── generate_plots.slurm
│
├── docs/
│   ├── design_summary.md           # One-page design rationale
│   └── urm_reverse_engineering.md  # Reverse-engineering of baseline URM
│
├── results/                        # Generated outputs (CSVs, plots) — not tracked
├── requirements.txt
├── setup_venv.sh
└── plan.md                         # Original implementation plan
```

---

## Setup

### Prerequisites

- Python 3.10+
- CUDA 11.8+ (for GPU training/inference)
- Access to a SLURM cluster with A100/H100 GPUs (for training)
- ~16GB GPU VRAM for inference, ~40GB for training

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/URM-MDN.git
cd URM-MDN
```

### 2. Create Virtual Environment

```bash
# Option A: Use the setup script
bash setup_venv.sh

# Option B: Manual setup
python -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### 3. Download the Baseline URM Model

The baseline model is loaded from HuggingFace. It will be downloaded automatically on first run, or you can pre-download it:

```bash
# Pre-download (optional, ~16GB)
python -c "from transformers import AutoModelForSequenceClassification; AutoModelForSequenceClassification.from_pretrained('LxzGorworworworworworworking/URM-LLaMa-3.1-8B')"
```

### 4. Verify Installation

```bash
# Quick sanity check (CPU, few samples)
PYTHONPATH=. python generate_comparison.py --device_map cpu --max_samples 10
```

---

## Training

Training is a 2-stage process with a frozen backbone throughout.

### Stage 1: Train MDN Attribute Head

Trains the MDN head on [HelpSteer2](https://huggingface.co/datasets/nvidia/HelpSteer2) attribute regression data.

```bash
# Label MDN (learns p(label | x)):
sbatch slurm/train_attribute_mdn.slurm

# Residual MDN (learns p(label − ŷ | x)) — recommended:
sbatch slurm/run_residual_train.slurm
```

Key arguments in `train_attribute_mdn.py`:
- `--uncertainty_target`: `"label"` or `"residual"` (default: `"label"`)
- `--num_components`: Number of mixture components K (default: 3)
- `--num_epochs`: Training epochs (default: 10)
- `--learning_rate`: Default 1e-4
- `--use_gaussian`: Use single Gaussian head instead of MDN (for ablation)

Checkpoints are saved to `checkpoints/stage1/` or `checkpoints/stage1_residual/`.

### Stage 2: Train Gating Network

Trains the attribute weighting network on [Skywork-Reward-80K](https://huggingface.co/datasets/Skywork/Skywork-Reward-Preference-80K-v0.1) using Bradley-Terry preference loss.

```bash
sbatch slurm/train_gating.slurm
```

Checkpoints are saved to `checkpoints/stage2/`.

---

## Evaluation

### Single Model on RewardBench

```bash
# Evaluate a specific checkpoint
PYTHONPATH=. python evaluate_rewardbench.py \
    --checkpoint_path checkpoints/stage2/best_model.pt \
    --uncertainty_target residual
```

### Full Comparison (all models)

```bash
# On GPU (generates predictions + plots):
sbatch slurm/run_plugin_eval.slurm

# On CPU (if predictions are already cached):
PYTHONPATH=. python generate_comparison.py --device_map cpu
```

This generates:
- `results/comparison_table.csv` — accuracy, NLL, uncertainty metrics for all configurations
- `results/abstention_results.csv` — active abstention accuracy at each retention rate
- `results/plots/` — comparison plots, abstention curves, diagnostic visualizations

---

## Technical Details

### MDN Head Architecture

Each attribute outputs K=3 logistic mixture components:
- **π** (mixing weights): softmax over K logits
- **μ** (component means): unconstrained
- **s** (component scales): softplus + ε

Total parameters: `5 attributes × 3 components × 3 params × 4096 hidden = ~184K`

### Uncertainty Metrics

| Metric | Formula | Use |
|---|---|---|
| **Mixture variance** | `Var = Σ πᵢ(π²s²ᵢ/3 + (μᵢ − μ̄)²)` | Primary uncertainty signal |
| **MAD₀** | `Σ πᵢ · sᵢ · [softplus(μᵢ/sᵢ) + softplus(−μᵢ/sᵢ)]` | Robust alternative to variance |
| **Sequence uncertainty** | `U = Σⱼ wⱼ² · Varⱼ` | Aggregated across 5 attributes |
| **Risk-aware score** | `R − β · √U` | Score penalised by uncertainty |

### Label vs Residual Training Target

- **Label MDN**: Trains on raw attribute labels → captures aleatoric/data uncertainty → anti-correlated with errors (bad for abstention)
- **Residual MDN**: Trains on `(label − baseline_prediction)` → captures model error distribution → positively correlated with errors (good for abstention)

---

## Citation

If you use this code, please cite the original URM paper:

```bibtex
@article{urm2024,
  title={Uncertainty-aware Reward Model: Teaching Reward Models to Know What is Unknown},
  author={...},
  year={2024}
}
```

---

## License

This project is for research purposes. The baseline URM model follows its original license terms.


---

# RewardUQ: Uncertainty-Aware Reward Modeling

RewardUQ is a research framework for training and evaluating **Uncertainty-Aware Reward Models** (URMs) based on Large Language Models. 

This repository currently implements and extends the **Residual Mixture Density Network (MDN)** approach on top of QLoRA-finetuned causal language models (such as LLaMa-3-8B).

## Features
- **Residual MDN Head**: Predicts a base scalar reward alongside an explicit uncertainty distribution parameterization (means, variances, mixture weights) across multiple attributes (e.g. helpfulness, correctness, coherence, complexity, verbosity).
- **QLoRA Integration**: Leverages HuggingFace `peft` and `bitsandbytes` to efficiently train full reward architectures natively in 4-bit precision without OOMing on a single 12GB GPU.
- **Robust Pipeline**: Includes a unified data pipeline that explicitly bypasses common TRL tokenization bugs, cleanly mapping HelpSteer attribute annotations directly into custom PyTorch loss components.

## Getting Started

### 1. Installation
First, ensure you have your virtual environment activated, then install the local package:
```bash
pip install -e .
```
> **Note**: DeepSpeed is deliberately not used for single-GPU QLoRA runs in this pipeline to prevent Triton kernel compilation failures. Do not install it unless you are scaling to multi-GPU clusters.

### 2. Dataset
The framework expects pairs of annotated completions. Ensure your dataset is formatted as JSONL and placed at `data/joined_pairs.jsonl`. 

### 3. Training
You can launch the training job using SLURM. The wrapper script is located in your workspace root (`slurm_qlora.sh`). 

To run it locally:
```bash
python ../train_qlora.py
```
This script automatically applies a 4-bit NormalFloat configuration and pre-tokenizes the input responses before initializing the custom `ResidualMDNTrainer`.

### 4. Evaluation
Check the `slurm/evaluate_residual.slurm` scripts for running evaluation benchmarks on your trained checkpoints.

## Recent Fixes
- Added robust dataset handling to prevent `input_ids_chosen` collisions in older TRL versions.
- Disabled `trl` implicit renaming to safely feed `r1`/`r2` annotations to the Custom MDN Collator.
- Integrated `BitsAndBytesConfig` natively into pipeline initialization to fix `CUDA out of memory` errors on 12GB partitions.

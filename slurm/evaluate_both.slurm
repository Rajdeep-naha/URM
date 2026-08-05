#!/bin/bash
#SBATCH --job-name=eval_both
#SBATCH --output=/localstorage/home/f20221218/URM/results/slurm/eval_both_%j.out
#SBATCH --error=/localstorage/home/f20221218/URM/results/slurm/eval_both_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=h100-mig
#SBATCH --account=students-limited
#SBATCH --qos=student-1mig-limited
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --gres=gpu:nvidia_h100_nvl_1g.12gb:1

set -euo pipefail

PROJECT_ROOT="/localstorage/home/f20221218/URM"
VENV_PYTHON="$PROJECT_ROOT/venv/bin/python"

mkdir -p "$PROJECT_ROOT/results/slurm"
cd "$PROJECT_ROOT"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

export HF_HOME="$PROJECT_ROOT/.cache/huggingface"
export HF_DATASETS_CACHE="$PROJECT_ROOT/.cache/huggingface/datasets"
export HF_TOKEN="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"
export WANDB_MODE=offline
export MPLCONFIGDIR="/tmp/matplotlib-eval"

echo "=== Running Residual MDN Evaluation ==="
"$VENV_PYTHON" evaluate_rewardbench.py \
  --model_name_or_path /localstorage/home/f20221218/URM-LLaMa-3.1-8B \
  --mdn_head_weights checkpoints/stage1/best_mdn_head_residual.pt \
  --gating_weights checkpoints/residual_gating/best_gating_weights.pt \
  --save_dir results/rewardbench/residual \
  --gating_variant residual

echo "=== Running Gaussian MDN Evaluation ==="
"$VENV_PYTHON" evaluate_rewardbench.py \
  --model_name_or_path /localstorage/home/f20221218/URM-LLaMa-3.1-8B \
  --mdn_head_weights checkpoints/stage1/best_mdn_head_label.pt \
  --gating_weights checkpoints/stage2/best_gating_weights.pt \
  --save_dir results/rewardbench/gaussian \
  --gating_variant gaussian

echo "Job finished on $(date)"

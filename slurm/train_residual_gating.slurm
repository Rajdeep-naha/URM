#!/bin/bash
#SBATCH --job-name=train_res_gate
#SBATCH --output=/localstorage/home/f20221218/URM/results/slurm/train_res_gate_%j.out
#SBATCH --error=/localstorage/home/f20221218/URM/results/slurm/train_res_gate_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=h100-mig
#SBATCH --account=students-limited
#SBATCH --qos=student-1mig-limited
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
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

# Set WANDB API KEY directly since wandb needs it
export WANDB_API_KEY="0e008a9f30b925b410d54020aeb95627dd2c5029" 
export WANDB_PROJECT="mdn-urm-stage2"
export WANDB_MODE="offline"
export WANDB_DIR="$PROJECT_ROOT/results/wandb"
export WANDB_CONFIG_DIR="$PROJECT_ROOT/results/wandb/.config"
export WANDB_CACHE_DIR="$PROJECT_ROOT/results/wandb/.cache"
mkdir -p "$WANDB_DIR" "$WANDB_CONFIG_DIR" "$WANDB_CACHE_DIR"

echo "Running training script with Residual Gating..."
"$VENV_PYTHON" train_residual_gating.py --model_name_or_path /localstorage/home/f20221218/URM-LLaMa-3.1-8B --batch_size=1 --grad_accum_steps=32 --lr=1e-5 --epochs=1 --mdn_head_weights=checkpoints/stage1/best_mdn_head_residual.pt

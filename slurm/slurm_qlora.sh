#!/bin/bash
#SBATCH --job-name=qlora_mdn
#SBATCH --output=/localstorage/home/f20221218/URM/results/slurm/qlora_mdn_%j.out
#SBATCH --error=/localstorage/home/f20221218/URM/results/slurm/qlora_mdn_%j.err
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
VENV_PYTHON="/localstorage/home/f20221218/URM/venv/bin/python"

mkdir -p "/localstorage/home/f20221218/URM/results/slurm"
cd "$PROJECT_ROOT"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_ROOT}/rewarduq/src:${PYTHONPATH:-}"
export HF_HOME="/localstorage/home/f20221218/URM/.cache/huggingface"
export HF_DATASETS_CACHE="/localstorage/home/f20221218/URM/.cache/huggingface/datasets"
export HF_TOKEN="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"
export TRITON_CACHE_DIR="/localstorage/home/f20221218/URM/.cache/triton"
export WANDB_MODE=offline

echo "==========================================="
echo "Stage: Full QLoRA Residual MDN Training"
echo "Started on $(date)"
echo "Node: $(hostname)"
echo "==========================================="

# Check GPU
"$VENV_PYTHON" -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA devices: {torch.cuda.device_count()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo ""
echo "=== Running QLoRA Training ==="
"$VENV_PYTHON" train_qlora.py

echo "QLoRA MDN training finished successfully!"
echo "==========================================="

#!/bin/bash
#SBATCH --job-name=mdn_gate_train
#SBATCH --output=/localstorage/home/f20221218/URM/results/slurm/mdn_gate_%j.out
#SBATCH --error=/localstorage/home/f20221218/URM/results/slurm/mdn_gate_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=h100-full
#SBATCH --account=professors-limited
#SBATCH --qos=professor-fullgpu-limited
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --gres=gpu:nvidia_h100_nvl:1

# Note: The above resource configuration matches the H100 MIG example.

set -euo pipefail

PROJECT_ROOT="/localstorage/home/f20221218/URM"
VENV_PYTHON="$PROJECT_ROOT/venv/bin/python"

# Load HF token if available
if [ -f "$HOME/.hftoken" ]; then
    source "$HOME/.hftoken"
fi

mkdir -p "$PROJECT_ROOT/results/slurm"
cd "$PROJECT_ROOT"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"
export HF_DATASETS_CACHE="$PROJECT_ROOT/.cache/huggingface/datasets"
export HF_TOKEN="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"
export WANDB_MODE=offline

echo "Job started on $(date)"
echo "Running on node: $(hostname)"
echo "Project root: $PROJECT_ROOT"
echo "Python: $VENV_PYTHON"

echo ""
echo "=== Checking/Installing Dependencies ==="
# Dependencies are pre-installed in the virtual environment.
"$VENV_PYTHON" -m pip install --quiet -r requirements.txt

# Check GPU
echo "=== Checking GPU availability ==="
"$VENV_PYTHON" -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA devices: {torch.cuda.device_count()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo ""
echo "=== Launching Stage 2: Gating Network Training ==="
# Gating network training uses Bradley-Terry Loss on Skywork preference dataset
"$VENV_PYTHON" train_gating.py \
  --model_name_or_path /localstorage/home/f20221218/URM-LLaMa-3.1-8B \
  --mdn_head_weights checkpoints/stage1/best_mdn_head.pt \
  --batch_size 16 \
  --epochs 1 \
  --lr 1e-5 \
  --grad_accum_steps 1 \
  --save_dir checkpoints/stage2 \
  --wandb \
  "$@"

echo "Job finished on $(date)"

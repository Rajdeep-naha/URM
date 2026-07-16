#!/bin/bash
#SBATCH --job-name=mdn_residual_train
#SBATCH --output=/localstorage/home/f20221218/URM/results/slurm/mdn_residual_train_%j.out
#SBATCH --error=/localstorage/home/f20221218/URM/results/slurm/mdn_residual_train_%j.err
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

echo "==========================================="
echo "Stage 1: Residual-Based MDN Training"
echo "Started on $(date)"
echo "Node: $(hostname)"
echo "==========================================="

# Check GPU
"$VENV_PYTHON" -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA devices: {torch.cuda.device_count()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# 1. Prepare label model checkpoint path
if [ -f "checkpoints/stage1/best_mdn_head.pt" ]; then
    echo "Backing up label-trained checkpoint to checkpoints/stage1/best_mdn_head_label.pt..."
    cp "checkpoints/stage1/best_mdn_head.pt" "checkpoints/stage1/best_mdn_head_label.pt"
fi

echo ""
echo "=== Running Residual MDN Training ==="
"$VENV_PYTHON" train_attribute_mdn.py \
  --model_name_or_path /localstorage/home/f20221218/URM-LLaMa-3.1-8B \
  --uncertainty_target residual \
  --lambda_nll 1.0 \
  --epochs 10 \
  --lr 1e-3 \
  --save_dir checkpoints/stage1_residual \
  --batch_size 2 \
  --grad_accum_steps 8

echo ""
echo "=== Copying trained residual checkpoint to checkpoints/stage1/best_mdn_head_residual.pt ==="
mkdir -p checkpoints/stage1
cp checkpoints/stage1_residual/best_mdn_head.pt checkpoints/stage1/best_mdn_head_residual.pt

echo ""
echo "=== Copying validation diagnostic plots to artifacts directory ==="
mkdir -p /home/f20221218/.gemini/antigravity-ide/brain/16ede877-14ac-4ab7-beec-d4d786001061/
cp checkpoints/stage1_residual/plots/*.png /home/f20221218/.gemini/antigravity-ide/brain/16ede877-14ac-4ab7-beec-d4d786001061/ || echo "No stage1 plots found"

echo "Residual MDN training finished successfully!"
echo "==========================================="

#!/bin/bash
#SBATCH --job-name=mdn_plugin_eval
#SBATCH --output=/localstorage/home/f20221218/URM/results/slurm/mdn_plugin_eval_%j.out
#SBATCH --error=/localstorage/home/f20221218/URM/results/slurm/mdn_plugin_eval_%j.err
#SBATCH --time=12:00:00
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
echo "MDN Plug-in Evaluation on RewardBench"
echo "Started on $(date)"
echo "Node: $(hostname)"
echo "==========================================="

# Check GPU
"$VENV_PYTHON" -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA devices: {torch.cuda.device_count()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Make sure we clean up the previous cached files first to ensure fresh evaluations
echo "Removing cached predictions to run fresh evaluation..."
rm -f results/baseline_predictions.pt results/label_predictions.pt results/residual_predictions.pt

echo ""
echo "=== Running Master Comparison Script ==="
"$VENV_PYTHON" generate_comparison.py \
  --model_name_or_path /localstorage/home/f20221218/URM-LLaMa-3.1-8B \
  --label_mdn_head_weights checkpoints/stage1/best_mdn_head_label.pt \
  --residual_mdn_head_weights checkpoints/stage1/best_mdn_head_residual.pt \
  --gating_weights checkpoints/stage2/best_gating_weights.pt \
  --save_dir results \
  --batch_size 4

echo ""
echo "=== Copying generated comparison plots to artifacts directory ==="
mkdir -p /home/f20221218/.gemini/antigravity-ide/brain/16ede877-14ac-4ab7-beec-d4d786001061/
cp results/plots/*.png /home/f20221218/.gemini/antigravity-ide/brain/16ede877-14ac-4ab7-beec-d4d786001061/ || echo "No comparison plots found"

echo "==========================================="
echo "Job Finished on $(date)"
echo "==========================================="

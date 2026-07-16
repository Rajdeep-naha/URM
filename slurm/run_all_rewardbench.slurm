#!/bin/bash
#SBATCH --job-name=mdn_full_eval
#SBATCH --output=/localstorage/home/f20221218/URM/results/slurm/mdn_full_eval_%j.out
#SBATCH --error=/localstorage/home/f20221218/URM/results/slurm/mdn_full_eval_%j.err
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
mkdir -p "$PROJECT_ROOT/results/rewardbench/baseline"
cd "$PROJECT_ROOT"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"
export HF_DATASETS_CACHE="$PROJECT_ROOT/.cache/huggingface/datasets"
export HF_TOKEN="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"
export WANDB_MODE=offline

echo "==========================================="
echo "Full RewardBench Evaluation & Plotting Job"
echo "Started on $(date)"
echo "Node: $(hostname)"
echo "==========================================="

# Check GPU
"$VENV_PYTHON" -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA devices: {torch.cuda.device_count()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo ""
echo "Step 1: Evaluating Baseline URM Model on RewardBench..."
"$VENV_PYTHON" evaluate_rewardbench.py \
  --model_name_or_path /localstorage/home/f20221218/URM-LLaMa-3.1-8B \
  --dataset_name allenai/reward-bench \
  --dataset_split filtered \
  --batch_size 4 \
  --baseline \
  --save_dir results/rewardbench/baseline

echo ""
echo "Step 2: Evaluating MDN-URM Model on RewardBench..."
"$VENV_PYTHON" evaluate_rewardbench.py \
  --model_name_or_path /localstorage/home/f20221218/URM-LLaMa-3.1-8B \
  --dataset_name allenai/reward-bench \
  --dataset_split filtered \
  --batch_size 4 \
  --mdn_head_weights checkpoints/stage1/best_mdn_head.pt \
  --gating_weights checkpoints/stage2/best_gating_weights.pt \
  --save_dir results/rewardbench

echo ""
echo "Step 3: Generating Comparison Plots and Diagnostics..."
"$VENV_PYTHON" generate_plots.py

echo ""
echo "Step 4: Copying generated plots to artifacts directory..."
cp results/plots/*.png /home/f20221218/.gemini/antigravity-ide/brain/16ede877-14ac-4ab7-beec-d4d786001061/

echo "==========================================="
echo "Job Finished on $(date)"
echo "==========================================="

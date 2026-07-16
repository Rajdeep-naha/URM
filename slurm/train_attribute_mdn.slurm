#!/bin/bash
#SBATCH --job-name=mdn_attr_train
#SBATCH --output=/localstorage/home/f20221218/URM/results/slurm/mdn_attr_%j.out
#SBATCH --error=/localstorage/home/f20221218/URM/results/slurm/mdn_attr_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=h100-full
#SBATCH --account=professors-limited
#SBATCH --qos=professor-fullgpu-limited
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --gres=gpu:nvidia_h100_nvl:1

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
# We run a quick check/sync to ensure everything is correct.
"$VENV_PYTHON" -m pip install --quiet -r requirements.txt

# Check GPU
echo "=== Checking GPU availability ==="
"$VENV_PYTHON" -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA devices: {torch.cuda.device_count()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo ""
echo "=== Launching Stage 1: MDN Attribute Regression ==="
# Launch with accelerate on a single GPU
"$PROJECT_ROOT/venv/bin/accelerate" launch \
  --num_machines 1 \
  --num_processes 1 \
  --mixed_precision bf16 \
  train_attribute_mdn.py \
  --model_name_or_path /localstorage/home/f20221218/URM-LLaMa-3.1-8B \
  --batch_size 2 \
  --epochs 10 \
  --lr 1e-3 \
  --grad_accum_steps 8 \
  --save_dir checkpoints/stage1 \
  --wandb \
  "$@"

echo "Job finished on $(date)"

#!/bin/bash
#SBATCH --job-name=gen_baseline
#SBATCH --output=/localstorage/home/f20221218/URM/results/slurm/gen_baseline_%j.out
#SBATCH --error=/localstorage/home/f20221218/URM/results/slurm/gen_baseline_%j.err
#SBATCH --time=01:00:00
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

echo "Job started on $(date)"

# Run generate_comparison.py for baseline
"$VENV_PYTHON" generate_comparison.py

echo "Job finished on $(date)"

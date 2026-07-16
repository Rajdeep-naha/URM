#!/bin/bash
set -euo pipefail

VENV_PATH="/home/f20221218/URM/venv"

if [ -d "$VENV_PATH" ] && [ ! -f "$VENV_PATH/bin/activate" ]; then
    echo "Found incomplete virtual environment at $VENV_PATH. Removing it..."
    rm -rf "$VENV_PATH"
fi

if [ ! -d "$VENV_PATH" ]; then
    echo "Creating virtual environment at $VENV_PATH using virtualenv..."
    virtualenv "$VENV_PATH"
fi

echo "Activating virtual environment..."
source "$VENV_PATH/bin/activate"

echo "Upgrading pip, setuptools, wheel..."
pip install --upgrade pip setuptools wheel

echo "Installing PyTorch (GPU CUDA version for development)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo "Installing transformers, datasets, accelerate, tqdm, wandb, matplotlib, scikit-learn, deepspeed..."
pip install transformers datasets accelerate tqdm sentencepiece tiktoken protobuf wandb matplotlib scikit-learn deepspeed

echo "Virtual environment setup complete!"

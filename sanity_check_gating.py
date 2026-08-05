import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, LlamaConfig
from models.modeling_mdn_urm import LlamaForSequenceClassificationWithMDN
from models.residual_gating import ResidualGating

def main():
    gating_weights = "checkpoints/stage2/best_gating_weights.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Initializing standalone gating networks for sanity check...")
    from models.modeling_mdn_urm import Weights
    
    # 4096 is the hidden size for LLaMa-3.1-8B
    model_orig = Weights(hidden_size=4096).to(device)
    model_res = ResidualGating(hidden_size=4096).to(device)
    
    print(f"Loading weights from {gating_weights}...")
    state_dict = torch.load(gating_weights, map_location=device)
    
    # We need to map the keys because the saved state dict might have 'fc.' prefixes
    # depending on how it was saved. Let's see what keys are there.
    # Actually, the saved state dict in train_gating.py saves `model.weights.state_dict()`.
    # So the keys should perfectly match `Weights` and `ResidualGating`.
    model_orig.load_state_dict(state_dict)
    model_res.load_state_dict(state_dict)
    
    model_orig.eval()
    model_res.eval()

    print("Generating random input of shape [Batch=2, Seq=64, Hidden=4096]...")
    x = torch.randn(2, 64, 4096, dtype=torch.bfloat16).to(device)

    with torch.no_grad():
        out_orig = model_orig(x)
        out_res = model_res(x)

    print("Comparing outputs...")
    diff = (out_orig - out_res).abs().max().item()
    
    print(f"Max difference in outputs: {diff}")
    
    if diff < 1e-6:
        print("Sanity Check Passed! The outputs are bitwise identical.")
    else:
        print("Sanity Check Failed!")

if __name__ == "__main__":
    main()

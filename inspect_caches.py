"""
Inspect keys and shapes of cached prediction files.
"""
import torch

for name in ["baseline", "label", "residual"]:
    path = f"results/{name}_predictions.pt"
    try:
        data = torch.load(path, map_location="cpu")
        print(f"File: {path}")
        print(f"  Keys: {list(data.keys())}")
        for k in ["chosen_scores", "rejected_scores", "chosen_sigmas", "chosen_weights"]:
            if k in data:
                val = data[k]
                if isinstance(val, torch.Tensor):
                    print(f"  {k}: shape {val.shape}, dtype {val.dtype}")
                elif isinstance(val, list):
                    print(f"  {k}: list of length {len(val)}")
    except Exception as e:
        print(f"Error loading {path}: {e}")

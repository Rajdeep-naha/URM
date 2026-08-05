import torch
state_dict = torch.load("checkpoints/stage1/best_mdn_head_residual.pt", map_location="cpu")
print("Keys in best_mdn_head_residual.pt:")
for k in state_dict.keys():
    print("  ", k, state_dict[k].shape)
print("\nKeys in best_gating_weights.pt:")
state_dict2 = torch.load("checkpoints/residual_gating/best_gating_weights.pt", map_location="cpu")
for k in state_dict2.keys():
    print("  ", k, state_dict2[k].shape)

import torch
import torch.nn as nn

class ResidualGating(nn.Module):
    """
    Separate Gating network matching original URM gating layers.
    This is used to train a clean, independent gating network for the Residual MDN
    without overwriting the original weights.
    """
    def __init__(self, hidden_size=4096):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SELU(),
            nn.Linear(hidden_size, 5)
        )

    def forward(self, x):
        # Cast input to match the linear layers' weight dtype dynamically
        return self.fc(x.to(self.fc[0].weight.dtype))

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.distribution_statistics import mixture_mean, mixture_variance

NUM_ATTRIBUTES = 5
NUM_COMPONENTS = 3


class URMMDNHead(nn.Module):
    """
    MDN Attribute Head for URM.
    Replaces 5 attributes x (mu, sigma) with 5 attributes x K mixture components.
    Supports either 'label' target training or 'residual' target training.
    """
    def __init__(
        self,
        hidden_size=4096,
        num_attributes=NUM_ATTRIBUTES,
        num_components=NUM_COMPONENTS,
        use_gaussian=False,
        uncertainty_target="label"
    ):
        super().__init__()
        self.num_attributes = num_attributes
        self.num_components = num_components
        self.use_gaussian = use_gaussian
        self.uncertainty_target = uncertainty_target
        
        if use_gaussian:
            self.output_dim = num_attributes * 2
        else:
            self.output_dim = num_attributes * num_components * 3

        self.proj = nn.Linear(hidden_size, self.output_dim)
        
        if self.uncertainty_target == "residual":
            self.mean_proj = nn.Linear(hidden_size, num_attributes)
            # Xavier Uniform initialization for weights, zero initialization for bias
            nn.init.xavier_uniform_(self.mean_proj.weight)
            nn.init.zeros_(self.mean_proj.bias)
            
        # Xavier Uniform initialization for weights, zero initialization for bias
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        
        # Bias logistic scale or gaussian std components slightly positive (e.g. 0.5) to avoid near-zero scales
        with torch.no_grad():
            if use_gaussian:
                bias_reshaped = self.proj.bias.view(self.num_attributes, 2)
                bias_reshaped[:, 1].fill_(0.5)
            else:
                bias_reshaped = self.proj.bias.view(self.num_attributes, 3, self.num_components)
                bias_reshaped[:, 2, :].fill_(0.5)

    def forward(self, hidden_states):
        """
        hidden_states: [B, L, hidden_size]
        returns:
            If uncertainty_target == "residual":
                mean_out: [B, L, 5] (predicted expected reward)
                And residual distribution parameters:
                pi, mu, s or mu, sigma
            Else (uncertainty_target == "label"):
                If use_gaussian is True:
                    mu: [B, L, 5]
                    sigma: [B, L, 5]
                Else:
                    pi: [B, L, 5, 3]
                    mu: [B, L, 5, 3]
                    s:  [B, L, 5, 3]
        """
        out = self.proj(hidden_states)
        B, L, _ = out.shape

        if self.use_gaussian:
            out = out.view(B, L, self.num_attributes, 2)
            mu = out[:, :, :, 0]
            raw_sigma = out[:, :, :, 1]
            sigma = F.softplus(raw_sigma) + 1e-6
            
            if self.uncertainty_target == "residual":
                mean_out = self.mean_proj(hidden_states)
                return mean_out, mu, sigma
            else:
                return mu, sigma
        else:
            # Reshape to [B, L, num_attributes, 3 (pi, mu, s), num_components]
            out = out.view(
                B,
                L,
                self.num_attributes,
                3,
                self.num_components
            )

            logits_pi = out[:, :, :, 0, :]
            mu = out[:, :, :, 1, :]
            raw_s = out[:, :, :, 2, :]

            pi = F.softmax(logits_pi, dim=-1)
            s = F.softplus(raw_s) + 1e-6

            if self.uncertainty_target == "residual":
                mean_out = self.mean_proj(hidden_states)
                return mean_out, pi, mu, s
            else:
                return pi, mu, s


def log_mixture_pdf(r, pi, mu, s):
    """
    Log density log p(r | x) under the Logistic mixture.
    r:  [...] (ground-truth targets, e.g. [B, L, 5] or [B, 5])
    pi: [..., K] (mixture weights)
    mu: [..., K] (component means)
    s:  [..., K] (component scales)
    returns: [...]
    """
    r = r.unsqueeze(-1)  # shape: [..., 1]
    z = (r - mu) / s

    # Log PDF of standard Logistic component: -z - 2*softplus(-z) - log(s)
    log_pdf = -torch.log(s) - F.softplus(z) - F.softplus(-z)  # shape: [..., K]
    log_pi = torch.log(pi + 1e-10)  # shape: [..., K]

    return torch.logsumexp(log_pi + log_pdf, dim=-1)


def mdn_nll_loss(target, pi, mu, s, attention_mask=None):
    """
    Negative Log Likelihood loss for the mixture.
    target: [B, L, 5] or [B, 5]
    pi, mu, s: mixture parameters matching targets
    attention_mask: [B, L] (optional mask to ignore padded tokens)
    """
    log_prob = log_mixture_pdf(target, pi, mu, s)  # shape: [B, L, 5] or [B, 5]

    if attention_mask is not None:
        mask = attention_mask.unsqueeze(-1).expand_as(log_prob)
        loss = -log_prob * mask
        return loss.sum() / (mask.sum() + 1e-8)
    else:
        return -log_prob.mean()


def gaussian_nll_loss(target, mu, sigma, attention_mask=None):
    """
    Negative Log Likelihood loss for the Gaussian head baseline.
    target: [B, L, 5] or [B, 5]
    mu, sigma: parameters of the Gaussian components
    attention_mask: [B, L] (optional mask)
    """
    import math
    variance = sigma ** 2
    log_prob = -0.5 * torch.log(2 * math.pi * variance) - ((target - mu) ** 2) / (2 * variance)

    if attention_mask is not None:
        mask = attention_mask.unsqueeze(-1).expand_as(log_prob)
        loss = -log_prob * mask
        return loss.sum() / (mask.sum() + 1e-8)
    else:
        return -log_prob.mean()

import torch
import torch.nn.functional as F
import math

def mixture_mean(pi, mu):
    """
    Calculate the expected reward (mean) of the mixture.
    pi: [..., K] (mixture weights)
    mu: [..., K] (component means)
    returns: [...]
    """
    return torch.sum(pi * mu, dim=-1)


def mixture_variance_decomposition(pi, mu, s):
    """
    Calculate the within-component and between-component variance decomposition.
    Within-component variance: sum_i pi_i * (pi^2 / 3) * s_i^2
    Between-component variance: sum_i pi_i * (mu_i - mu_mix)^2
    """
    mean = mixture_mean(pi, mu)
    
    # Within-component variance
    logistic_var = (math.pi ** 2 / 3.0) * (s ** 2)
    within_var = torch.sum(pi * logistic_var, dim=-1)
    
    # Between-component variance
    # mu has shape [..., K], mean has shape [...]
    # Unsqueeze mean to align with mu's K dimension
    mean_expanded = mean.unsqueeze(-1)
    between_var = torch.sum(pi * ((mu - mean_expanded) ** 2), dim=-1)
    
    return within_var, between_var


def mixture_variance(pi, mu, s):
    """
    Calculate the total variance of the mixture (within + between).
    """
    within_var, between_var = mixture_variance_decomposition(pi, mu, s)
    return within_var + between_var


def mixture_std(pi, mu, s):
    """
    Calculate the standard deviation of the mixture.
    """
    return torch.sqrt(torch.clamp(mixture_variance(pi, mu, s), min=1e-12))


def mixture_mad0(pi, mu, s):
    """
    Calculate the Mean Absolute Deviation around Zero (MAD_0) of the mixture.
    MAD_0 = sum_i pi_i * s_i * [log(1 + e^(mu_i/s_i)) + log(1 + e^(-mu_i/s_i))]
    """
    ratio = mu / torch.clamp(s, min=1e-12)
    # Numerically stable log(1 + e^x) + log(1 + e^-x) using F.softplus
    bracket = F.softplus(ratio) + F.softplus(-ratio)
    comp_mad0 = s * bracket
    return torch.sum(pi * comp_mad0, dim=-1)


def mixture_entropy(pi):
    """
    Calculate the Shannon entropy of the mixture weights pi.
    H(pi) = -sum_i pi_i * log(pi_i)
    """
    return -torch.sum(pi * torch.log(pi + 1e-12), dim=-1)


def compute_statistics(pi, mu, s):
    """
    Calculate and return all analytical uncertainty statistics as a dictionary.
    """
    mean = mixture_mean(pi, mu)
    within_var, between_var = mixture_variance_decomposition(pi, mu, s)
    total_var = within_var + between_var
    std = torch.sqrt(torch.clamp(total_var, min=1e-12))
    mad0 = mixture_mad0(pi, mu, s)
    entropy = mixture_entropy(pi)
    return {
        "mean": mean,
        "variance": total_var,
        "within_variance": within_var,
        "between_variance": between_var,
        "std": std,
        "mad0": mad0,
        "entropy": entropy
    }

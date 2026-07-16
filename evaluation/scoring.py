import torch

def get_urm_score(expected_rewards, weights):
    """
    Original URM expected reward calculation:
    R = sum_j w_j * E[r_j]
    expected_rewards: [B, 5]
    weights: [B, 5]
    returns: [B]
    """
    return (expected_rewards * weights).sum(dim=-1)


def get_urm_uncertainty(variances, weights):
    """
    Sequence-level reward variance:
    U = sum_j w_j^2 * Var(r_j)
    variances: [B, 5]
    weights: [B, 5]
    returns: [B]
    """
    return ((weights ** 2) * variances).sum(dim=-1)


def get_urm_std(variances, weights):
    """
    Sequence-level standard deviation:
    SD = sqrt(sum_j w_j^2 * Var(r_j))
    variances: [B, 5]
    weights: [B, 5]
    returns: [B]
    """
    var = get_urm_uncertainty(variances, weights)
    return torch.sqrt(torch.clamp(var, min=1e-12))


def get_mad_uncertainty(mads, weights):
    """
    Sequence-level Mean Absolute Deviation (MAD):
    MAD = sum_j |w_j| * MAD_{0, j}
    mads: [B, 5]
    weights: [B, 5]
    returns: [B]
    """
    return (torch.abs(weights) * mads).sum(dim=-1)


def risk_aware_variance_score(reward, variance, beta):
    """
    Downstream reward correction using standard deviation:
    R_corr = R - beta * SD
    where SD = sqrt(variance)
    """
    sd = torch.sqrt(torch.clamp(variance, min=1e-12))
    return reward - beta * sd


def risk_aware_variance_only_score(reward, variance, beta):
    """
    Downstream reward correction using raw variance:
    R_corr = R - beta * Var
    """
    return reward - beta * variance


def risk_aware_mad_score(reward, mad, beta):
    """
    Downstream reward correction using MAD:
    R_corr = R - beta * MAD
    """
    return reward - beta * mad

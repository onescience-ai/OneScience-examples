import torch


def inr_reconstruction_loss(pred, target):
    """L_mu = E[(u(x) - f(x))^2]."""
    return ((pred - target) ** 2).mean()


def code_mse_loss(pred_codes, target_codes):
    """MSE in z-code space (g_psi supervision)."""
    return ((pred_codes - target_codes) ** 2).mean()

import math

import torch

from .hadamard import get_hadamard_matrix


def get_phi_s_matrix(X: torch.Tensor):
    mean_val = X.mean(dim=0)

    X = X - mean_val

    cov = torch.cov(X.T)

    e, U = torch.linalg.eigh(cov)
    mask = e >= 0
    e = torch.where(mask, e, torch.zeros_like(e))

    H = get_hadamard_matrix(e.shape[0]).to(U.dtype)

    normalizer = e.mean().sqrt()
    inv_normalizer = 1. / normalizer

    whiten = inv_normalizer * (H @ U.T)

    return mean_val, whiten
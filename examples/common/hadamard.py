import math
import os

import torch


def get_hadamard_matrix(feature_dim: int, allow_approx: bool = True):
    H = _get_hadamard_matrix(feature_dim, allow_approx)

    n = H.norm(dim=1, keepdim=True)
    H /= n

    assert H.shape[0] == feature_dim and H.shape[1] == feature_dim, "Invalid Hadamard matrix!"
    f = H @ H.T
    assert torch.allclose(f, torch.eye(feature_dim, dtype=H.dtype, device=H.device)), "Invalid orthogonal construction!"

    return H


def _get_hadamard_matrix(feature_dim: int, allow_approx: bool):
    if feature_dim <= 0:
        raise ValueError(f'Invalid `feature_dim`. Must be a positive integer!')
    if feature_dim > 2 and feature_dim % 4 != 0:
        if allow_approx:
            return get_bernoulli_matrix(feature_dim)
        raise ValueError(f'`feature_dim` certainly needs to be divisible by 4, or be 1 or 2!')

    pw2 = math.log2(feature_dim)
    is_pow2 = int(pw2) == pw2

    if is_pow2:
        return get_sylvester_hadamard_matrix(feature_dim)

    # Not so simple anymore. We need to see if we can use Paley's construction
    prime_factors = _get_prime_factors(feature_dim)
    num_2s = sum(1 for v in prime_factors if v == 2)

    for i in range(num_2s - 1, -1, -1):
        syl_size = 2 ** i
        paley_size = 1
        for k in range(i, len(prime_factors)):
            paley_size *= prime_factors[k]

        # Paley Construction 1
        paley = None
        if _is_paley_construction_1(paley_size):
            paley = get_paley_hadamard_matrix_1(paley_size)
        elif _is_paley_construction_2(paley_size):
            paley = get_paley_hadamard_matrix_2(paley_size)

        if paley is not None:
            if syl_size > 1:
                syl = get_sylvester_hadamard_matrix(syl_size)
                return get_joint_hadamard_matrix(syl, paley)
            return paley

    if allow_approx:
        return get_bernoulli_matrix(feature_dim)
    raise ValueError(f'Unsupported `feature_dim`.')


def get_sylvester_hadamard_matrix(feature_dim: int):
    pw2 = math.log2(feature_dim)
    is_pow2 = int(pw2) == pw2
    if not is_pow2:
        raise ValueError("The `feature_dim` must be a power of 2 for this algorithm!")

    A = torch.ones(1, 1, dtype=torch.float64)
    while A.shape[0] < feature_dim:
        B = A.repeat(2, 2)
        B[-A.shape[0]:, -A.shape[0]:] *= -1

        A = B

    assert A.shape[0] == feature_dim, "Invalid algorithm!"

    return A


def get_paley_hadamard_matrix_1(feature_dim: int):
    q = feature_dim - 1
    Q = _get_paley_q(q)

    H = torch.eye(feature_dim, dtype=torch.float32)
    H[0, 1:].fill_(1)
    H[1:, 0].fill_(-1)
    H[1:, 1:] += Q

    return H


def get_paley_hadamard_matrix_2(feature_dim: int):
    q = feature_dim // 2 - 1
    Q = _get_paley_q(q)

    inner = torch.zeros(q + 1, q + 1, dtype=Q.dtype, device=Q.device)
    inner[0, 1:].fill_(1)
    inner[1:, 0].fill_(1)
    inner[1:, 1:].copy_(Q)

    zero_cells = torch.tensor([
        [1, -1],
        [-1, -1],
    ], dtype=inner.dtype, device=inner.device)

    pos_cells = torch.tensor([
        [1, 1],
        [1, -1],
    ], dtype=inner.dtype, device=inner.device)
    neg_cells = -pos_cells

    full_zero = zero_cells.repeat(*inner.shape)
    full_pos = pos_cells.repeat(*inner.shape)
    full_neg = neg_cells.repeat(*inner.shape)

    full_inner = inner.repeat_interleave(2, dim=0).repeat_interleave(2, dim=1)

    full_zero = torch.where(full_inner == 0, full_zero, 0)
    full_pos = torch.where(full_inner > 0, full_pos, 0)
    full_neg = torch.where(full_inner < 0, full_neg, 0)

    H = full_zero + full_pos + full_neg

    return H


def _get_paley_q(q: int):
    b_opts = torch.arange(1, q, dtype=torch.float32).pow_(2) % q

    indexer = torch.arange(1, q + 1, dtype=torch.float32)
    m_indexer = indexer[:, None] - indexer[None]
    m_indexer = m_indexer % q

    is_zero = m_indexer == 0
    is_square = torch.any(m_indexer[..., None] == b_opts[None, None], dim=2)

    sq_vals = is_square.float().mul_(2).sub_(1)

    Q = torch.where(is_zero, 0, sq_vals)
    return Q


def get_joint_hadamard_matrix(syl: torch.Tensor, paley: torch.Tensor):
    ret = torch.kron(syl, paley)
    return ret


def get_bernoulli_matrix(feature_dim: int):
    A = (torch.rand(feature_dim, feature_dim, dtype=torch.float32) > 0.5).double().mul_(2).sub_(1)

    Q, _ = torch.linalg.qr(A)

    Q = torch.where((Q.diag() > 0)[None], Q, -Q)

    Q = Q.T.contiguous()
    return Q


def _is_paley_construction(q: int, modulo: int):
    is_paley = False
    if _is_prime(q):
        for z in range(1, 11):
            qz = q ** z
            if qz % 4 == modulo:
                is_paley = True
                break
    return is_paley


def _is_paley_construction_1(feature_dim: int):
    q = feature_dim - 1
    return _is_paley_construction(q, modulo=3)


def _is_paley_construction_2(feature_dim: int):
    q = feature_dim // 2 - 1
    return _is_paley_construction(q, modulo=1)


def _is_prime(n: int):
    factors = _get_prime_factors(n)
    return len(factors) == 1


def _get_prime_factors(n: int):
    i = 2
    factors = []
    while i * i <= n:
        if n % i != 0:
            i += 1
        else:
            n //= i
            factors.append(i)
    if n > 1:
        factors.append(n)
    return factors

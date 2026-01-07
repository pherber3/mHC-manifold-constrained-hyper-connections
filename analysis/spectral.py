"""Spectral analysis utilities for doubly stochastic matrices."""

import torch
from typing import Dict


def compute_eigenvalues(H_res: torch.Tensor) -> torch.Tensor:
    """Return eigenvalues sorted by magnitude (descending).

    Args:
        H_res: (n, n) matrix

    Returns:
        Tensor of eigenvalue magnitudes sorted descending
    """
    eigs = torch.linalg.eigvals(H_res)
    return eigs.abs().sort(descending=True).values


def spectral_gap(H_res: torch.Tensor) -> float:
    """Compute spectral gap: 1 - |lambda_2|.

    The spectral gap controls mixing rate in Markov chains.
    - Gap near 0: slow mixing (near permutation matrix)
    - Gap near 1: fast mixing (near uniform matrix)

    Args:
        H_res: (n, n) doubly stochastic matrix

    Returns:
        Spectral gap value in [0, 1]
    """
    eigs = compute_eigenvalues(H_res)
    return 1.0 - eigs[1].item()


def analyze_h_res(H_res: torch.Tensor) -> Dict[str, float]:
    """Full spectral analysis of a doubly stochastic matrix.

    Args:
        H_res: (n, n) doubly stochastic matrix

    Returns:
        Dictionary with spectral properties:
        - lambda_1: largest eigenvalue magnitude (should be 1.0)
        - lambda_2_abs: second-largest eigenvalue magnitude
        - spectral_gap: 1 - |lambda_2|
        - dist_to_uniform: Frobenius distance to (1/n)11^T
        - dist_to_identity: Frobenius distance to I
        - entropy: matrix entropy -sum(H * log(H))
        - min_entry, max_entry: entry bounds
    """
    eigs = compute_eigenvalues(H_res)
    n = H_res.shape[0]
    uniform = torch.ones_like(H_res) / n
    identity = torch.eye(n, device=H_res.device, dtype=H_res.dtype)

    # Compute entropy with numerical stability
    H_clamped = H_res.clamp(min=1e-10)
    entropy = -(H_res * H_clamped.log()).sum().item()

    return {
        "lambda_1": eigs[0].item(),
        "lambda_2_abs": eigs[1].item(),
        "spectral_gap": 1.0 - eigs[1].item(),
        "dist_to_uniform": (H_res - uniform).norm().item(),
        "dist_to_identity": (H_res - identity).norm().item(),
        "entropy": entropy,
        "min_entry": H_res.min().item(),
        "max_entry": H_res.max().item(),
    }


def mixing_time_estimate(H_res: torch.Tensor, epsilon: float = 0.01) -> float:
    """Estimate mixing time: t_mix ~ log(1/epsilon) / spectral_gap.

    Args:
        H_res: (n, n) doubly stochastic matrix
        epsilon: target distance from stationary distribution

    Returns:
        Estimated mixing time (number of matrix multiplications).
        Returns float('inf') if spectral gap is 0.
    """
    import math

    gap = spectral_gap(H_res)
    if gap < 1e-10:
        return float("inf")
    return math.log(1.0 / epsilon) / gap

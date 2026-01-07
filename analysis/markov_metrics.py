"""Markov chain mixing metrics for H_res analysis."""

import torch
from typing import Dict, List


def distance_to_uniform(H_res: torch.Tensor) -> float:
    """Frobenius norm distance to uniform matrix (1/n)11^T.

    Args:
        H_res: (n, n) matrix

    Returns:
        Frobenius distance to uniform matrix
    """
    n = H_res.shape[0]
    uniform = torch.ones_like(H_res) / n
    return (H_res - uniform).norm().item()


def distance_to_identity(H_res: torch.Tensor) -> float:
    """Frobenius norm distance to identity matrix.

    Args:
        H_res: (n, n) matrix

    Returns:
        Frobenius distance to identity
    """
    identity = torch.eye(H_res.shape[0], device=H_res.device, dtype=H_res.dtype)
    return (H_res - identity).norm().item()


def matrix_entropy(H_res: torch.Tensor, eps: float = 1e-10) -> float:
    """Compute matrix entropy: -sum(H * log(H)).

    Higher entropy means more uniform distribution of entries.

    Args:
        H_res: (n, n) non-negative matrix
        eps: small value for numerical stability

    Returns:
        Entropy value (higher = more uniform)
    """
    H_clamped = H_res.clamp(min=eps)
    return -(H_res * H_clamped.log()).sum().item()


def normalized_entropy(H_res: torch.Tensor, eps: float = 1e-10) -> float:
    """Compute entropy normalized by maximum possible entropy.

    Returns value in [0, 1] where 1 = uniform matrix.

    Args:
        H_res: (n, n) non-negative matrix
        eps: small value for numerical stability

    Returns:
        Normalized entropy in [0, 1]
    """
    import math

    n = H_res.shape[0]
    max_entropy = n * n * math.log(n)  # Entropy of uniform matrix
    return matrix_entropy(H_res, eps) / max_entropy


def cumulative_product_metrics(
    h_res_list: List[torch.Tensor],
) -> Dict[str, List[float]]:
    """Track mixing metrics for cumulative H_res products.

    Computes metrics for: H_1, H_1 @ H_2, H_1 @ H_2 @ H_3, ...

    This tests whether the residual path converges toward uniform
    as depth increases (which would indicate Markov mixing problem).

    Args:
        h_res_list: List of (n, n) doubly stochastic matrices

    Returns:
        Dictionary with lists of metrics at each depth:
        - dist_to_uniform: Frobenius distance to uniform
        - dist_to_identity: Frobenius distance to identity
        - spectral_gap: 1 - |lambda_2| of cumulative product
    """
    if not h_res_list:
        return {"dist_to_uniform": [], "dist_to_identity": [], "spectral_gap": []}

    n = h_res_list[0].shape[0]
    device = h_res_list[0].device
    dtype = h_res_list[0].dtype

    uniform = torch.ones(n, n, device=device, dtype=dtype) / n
    identity = torch.eye(n, device=device, dtype=dtype)

    product = identity.clone()
    metrics: Dict[str, List[float]] = {
        "dist_to_uniform": [],
        "dist_to_identity": [],
        "spectral_gap": [],
    }

    for H_res in h_res_list:
        product = H_res @ product

        metrics["dist_to_uniform"].append((product - uniform).norm().item())
        metrics["dist_to_identity"].append((product - identity).norm().item())

        eigs = torch.linalg.eigvals(product).abs().sort(descending=True).values
        metrics["spectral_gap"].append(1.0 - eigs[1].item())

    return metrics


def analyze_mixing_trajectory(
    h_res_list: List[torch.Tensor],
) -> Dict[str, float]:
    """Summarize mixing trajectory across layers.

    Args:
        h_res_list: List of (n, n) doubly stochastic matrices

    Returns:
        Summary statistics:
        - initial_dist_uniform: distance at layer 0
        - final_dist_uniform: distance at final layer
        - mixing_ratio: final/initial distance ratio (< 1 means converging)
        - max_spectral_gap: maximum spectral gap across products
        - avg_spectral_gap: average spectral gap
    """
    metrics = cumulative_product_metrics(h_res_list)

    if not metrics["dist_to_uniform"]:
        return {}

    dist_uniform = metrics["dist_to_uniform"]
    spectral_gaps = metrics["spectral_gap"]

    return {
        "initial_dist_uniform": dist_uniform[0],
        "final_dist_uniform": dist_uniform[-1],
        "mixing_ratio": dist_uniform[-1] / (dist_uniform[0] + 1e-10),
        "max_spectral_gap": max(spectral_gaps),
        "avg_spectral_gap": sum(spectral_gaps) / len(spectral_gaps),
    }

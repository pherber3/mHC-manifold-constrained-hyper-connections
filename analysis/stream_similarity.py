"""Stream similarity measurement via forward hooks."""

import torch
import torch.nn.functional as F
from torch import nn
from typing import List, Dict, Optional


def compute_stream_similarity(
    activations: torch.Tensor, num_streams: int
) -> torch.Tensor:
    """Compute pairwise cosine similarity between streams.

    Args:
        activations: (batch * num_streams, seq, dim) tensor
        num_streams: number of residual streams

    Returns:
        (num_streams, num_streams) similarity matrix averaged over batch
    """
    bs, seq, dim = activations.shape
    batch = bs // num_streams

    # Reshape: (batch, num_streams, seq, dim)
    x = activations.view(batch, num_streams, seq, dim)
    # Flatten seq and dim: (batch, num_streams, seq*dim)
    x = x.view(batch, num_streams, -1)
    # Normalize along feature dimension
    x = F.normalize(x, dim=-1)
    # Compute similarity: (batch, num_streams, num_streams)
    sim = torch.bmm(x, x.transpose(-1, -2))
    # Average over batch
    return sim.mean(dim=0)


def mean_off_diagonal_similarity(sim_matrix: torch.Tensor) -> float:
    """Compute mean off-diagonal similarity from similarity matrix.

    Args:
        sim_matrix: (num_streams, num_streams) similarity matrix

    Returns:
        Mean similarity between different streams (off-diagonal)
    """
    n = sim_matrix.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool, device=sim_matrix.device)
    return sim_matrix[mask].mean().item()


class StreamSimilarityTracker:
    """Track stream similarity across layers during forward pass.

    Usage:
        tracker = StreamSimilarityTracker(model, num_streams=4)
        with tracker:
            output = model(input)
        sims = tracker.get_mean_off_diagonal()

    The tracker registers hooks on each Block to capture activations
    and compute stream similarity.
    """

    def __init__(self, model: nn.Module, num_streams: int):
        """Initialize tracker.

        Args:
            model: GPT model with transformer.h containing Blocks
            num_streams: number of residual streams
        """
        self.model = model
        self.num_streams = num_streams
        self.similarities: List[torch.Tensor] = []
        self.handles: List[torch.utils.hooks.RemovableHandle] = []

    def __enter__(self) -> "StreamSimilarityTracker":
        """Register forward hooks on blocks."""
        self.similarities = []

        def hook(
            module: nn.Module, input: tuple, output: torch.Tensor
        ) -> None:
            sim = compute_stream_similarity(output.detach(), self.num_streams)
            self.similarities.append(sim)

        for block in self.model.transformer.h:
            handle = block.register_forward_hook(hook)
            self.handles.append(handle)

        return self

    def __exit__(self, *args) -> None:
        """Remove all hooks."""
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def get_similarity_matrices(self) -> List[torch.Tensor]:
        """Get raw similarity matrices for each layer.

        Returns:
            List of (num_streams, num_streams) similarity matrices
        """
        return self.similarities

    def get_mean_off_diagonal(self) -> List[float]:
        """Get mean off-diagonal similarity at each layer.

        Higher values indicate more stream homogenization.

        Returns:
            List of mean off-diagonal similarities per layer
        """
        return [mean_off_diagonal_similarity(sim) for sim in self.similarities]

    def get_stream_variance(self) -> List[float]:
        """Get variance across streams at each layer.

        Lower values indicate more stream homogenization.

        Returns:
            List of stream variances per layer
        """
        variances = []
        for sim in self.similarities:
            # Variance of off-diagonal elements
            n = sim.shape[0]
            mask = ~torch.eye(n, dtype=torch.bool, device=sim.device)
            variances.append(sim[mask].var().item())
        return variances


def forward_with_stream_similarity(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    num_streams: int,
    ctx: Optional[torch.amp.autocast] = None,
) -> tuple[torch.Tensor, List[float]]:
    """Run forward pass and measure stream similarity.

    Args:
        model: GPT model
        x: input tensor
        y: target tensor
        num_streams: number of residual streams
        ctx: optional autocast context

    Returns:
        Tuple of (loss, list of mean off-diagonal similarities per layer)
    """
    sims: List[float] = []
    handles: List[torch.utils.hooks.RemovableHandle] = []

    # Access raw model if wrapped in DDP
    raw_model = model.module if hasattr(model, "module") else model

    def hook(module: nn.Module, input: tuple, output: torch.Tensor) -> None:
        out = output.detach()
        bs, seq, dim = out.shape
        batch = bs // num_streams
        x_reshaped = out.view(batch, num_streams, seq, dim).view(
            batch, num_streams, -1
        )
        x_norm = F.normalize(x_reshaped, dim=-1)
        sim = torch.bmm(x_norm, x_norm.transpose(-1, -2)).mean(dim=0)
        mask = ~torch.eye(num_streams, dtype=torch.bool, device=sim.device)
        sims.append(sim[mask].mean().item())

    for block in raw_model.transformer.h:
        handles.append(block.register_forward_hook(hook))

    if ctx is not None:
        with ctx:
            _, loss = model(x, y)
    else:
        _, loss = model(x, y)

    for h in handles:
        h.remove()

    return loss, sims

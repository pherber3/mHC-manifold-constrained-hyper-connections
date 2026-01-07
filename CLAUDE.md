# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research implementation of **mHC (Manifold-Constrained Hyper-Connections)** from DeepSeek (arXiv:2512.24880) as a variant of Hyper-Connections (arXiv:2409.19606). This is a research prototype focused on correctness and clarity.

The core mHC layer update:
```
x_{l+1} = H_l^{res} x_l + H_l^{post,T} F(H_l^{pre} x_l, W_l)
```

Key constraints:
- `H_res`: doubly stochastic (Birkhoff polytope) via Sinkhorn-Knopp or orthostochastic via Newton-Schulz
- `H_pre`, `H_post`: non-negative mixing maps (softmax)

## Commands

### Running Tests
```bash
pytest                          # Run all tests
pytest tests/test_hyper_connections.py::test_mhc_H_res_constraints  # Single test
```

### Training (from examples/nanogpt/)
```bash
# 6-layer configs (~20M params)
python train.py config/train_fineweb10B.py       # Baseline
python train.py config/train_fineweb10B_hc.py    # Hyper-Connections
python train.py config/train_fineweb10B_mhc.py   # mHC
python train.py config/train_fineweb10B_vres.py  # Value residual

# 48-layer configs (~20M params)
python train.py config/train_fineweb10B_48l.py
python train.py config/train_fineweb10B_hc_48l.py
python train.py config/train_fineweb10B_mhc_48l.py
python train.py config/train_fineweb10B_vres_48l.py

# Multi-GPU
torchrun --standalone --nproc_per_node=4 train.py config/train_fineweb10B_mhc_48l.py
```

### Type Checking
```bash
basedpyright
```

## Architecture

### Core Library (`hyper_connections/`)

- **`hyper_connections.py`**: Main implementation containing:
  - `HyperConnections`: Core class supporting both standard HC and mHC modes (`mhc=True`)
  - `sinkhorn_log()`: Projects logits to doubly stochastic matrix via Sinkhorn-Knopp
  - `orthostochastic_project()`: Alternative projection using Newton-Schulz orthogonalization
  - `Residual`: Base class for simple residual connections
  - `get_init_and_expand_reduce_stream_functions()`: Factory for creating HC/mHC wrappers

- **`hyper_connections_mhc.py`**: Standalone mHC implementation (simpler, mHC-only)

### Key Integration Pattern

The library uses expand/reduce stream functions for multi-stream residuals:

```python
from hyper_connections import get_init_and_expand_reduce_stream_functions

init_hc, expand_stream, reduce_stream = get_init_and_expand_reduce_stream_functions(
    num_streams=4,    # Number of residual streams
    num_fracs=1,      # Fraction-connections (experimental)
    disable=False     # Set True to fallback to standard residuals
)

# In forward pass:
x = expand_stream(x)           # (B, ..., D) -> (B*S, ..., D)
x = hc_layer(x)                # Apply HC-wrapped layer
x = reduce_stream(x)           # (B*S, ..., D) -> (B, ..., D)
```

### mHC Configuration Options

When `mhc=True`:
- `sinkhorn_iters`, `sinkhorn_tau`: Sinkhorn projection params (default: 10, 0.05)
- `mhc_h_res_proj`: `"sinkhorn"` (default) or `"orthostochastic"`
- `ns_steps`, `ns_eps`, `ns_coeffs`: Newton-Schulz params for orthostochastic mode

### nanoGPT Example (`examples/nanogpt/`)

- **`model.py`**: GPT implementation with HC/mHC integration in `Block` class
- **`train.py`**: Training loop with W&B logging
- **`value_residual.py`**: Value residual state tracking for attention
- **`config/`**: Training configs (baseline, HC, mHC, value-residual variants)

### einops Notation

Used throughout codebase:
- `b` - batch
- `d` - feature dimension
- `s` - residual streams
- `t` - residual streams + num branch inputs
- `f` - number of fractions
- `v` - number of views for branch input

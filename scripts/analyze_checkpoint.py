#!/usr/bin/env python3
"""
Analyze Markov mixing properties of learned H_res matrices from a checkpoint.

Usage:
    python scripts/analyze_checkpoint.py --checkpoint out-mhc/ckpt.pt
    python scripts/analyze_checkpoint.py --checkpoint out-mhc/ckpt.pt --output analysis/
    python scripts/analyze_checkpoint.py --checkpoint out-mhc/ckpt.pt --projection orthostochastic
"""

import argparse
import json
from pathlib import Path
import sys

import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from hyper_connections.hyper_connections import sinkhorn_log, orthostochastic_project
from analysis.spectral import analyze_h_res
from analysis.markov_metrics import cumulative_product_metrics, analyze_mixing_trajectory


def load_checkpoint(path: str) -> dict:
    """Load checkpoint from path."""
    return torch.load(path, map_location="cpu", weights_only=False)


def extract_h_res_matrices(
    state_dict: dict,
    projection: str = "sinkhorn",
    sinkhorn_iters: int = 10,
    sinkhorn_tau: float = 0.05,
    ns_steps: int = 5,
    ns_eps: float = 1e-7,
    ns_coeffs: tuple = (3.0, -3.2, 1.2),
) -> list:
    """
    Extract H_res_logits from state dict and project to doubly stochastic.

    Keys follow pattern: transformer.h.{block_idx}.hc_{attn|mlp}.H_res_logits
    """
    h_res_keys = sorted([k for k in state_dict if "H_res_logits" in k])
    h_res_list = []

    for key in h_res_keys:
        logits = state_dict[key]
        if projection == "orthostochastic":
            H = orthostochastic_project(logits, ns_steps, ns_eps, ns_coeffs)
        else:
            H = sinkhorn_log(logits, sinkhorn_iters, sinkhorn_tau)
        h_res_list.append(H)

    return h_res_list


def run_analysis(
    checkpoint_path: str,
    projection: str = "sinkhorn",
    sinkhorn_iters: int = 10,
    sinkhorn_tau: float = 0.05,
) -> dict:
    """
    Run complete spectral analysis on a checkpoint.

    Returns dict with:
        - per_layer: Dict[int, Dict[str, float]] - metrics for each layer's H_res
        - cumulative: Dict[str, List[float]] - metrics for cumulative products
        - summary: Dict[str, float] - summary statistics
    """
    ckpt = load_checkpoint(checkpoint_path)
    state_dict = ckpt["model"]

    # Extract config if available
    config = ckpt.get("config", {})
    ns_steps = config.get("ns_steps", 5)
    ns_eps = config.get("ns_eps", 1e-7)
    ns_coeffs = config.get("ns_coeffs", (3.0, -3.2, 1.2))

    # Extract and project H_res matrices
    h_res_list = extract_h_res_matrices(
        state_dict,
        projection=projection,
        sinkhorn_iters=sinkhorn_iters,
        sinkhorn_tau=sinkhorn_tau,
        ns_steps=ns_steps,
        ns_eps=ns_eps,
        ns_coeffs=ns_coeffs,
    )

    if not h_res_list:
        return {"error": "No H_res_logits found in checkpoint"}

    # Per-layer analysis
    per_layer = {}
    for i, H in enumerate(h_res_list):
        per_layer[i] = analyze_h_res(H)

    # Cumulative product analysis
    cumulative = cumulative_product_metrics(h_res_list)

    # Summary statistics
    summary = analyze_mixing_trajectory(h_res_list)
    summary["num_layers"] = len(h_res_list)
    summary["num_streams"] = h_res_list[0].shape[0]

    # Average per-layer stats
    avg_lambda2 = sum(per_layer[i]["lambda_2_abs"] for i in per_layer) / len(per_layer)
    avg_spectral_gap = sum(per_layer[i]["spectral_gap"] for i in per_layer) / len(
        per_layer
    )
    summary["avg_lambda2_per_layer"] = avg_lambda2
    summary["avg_spectral_gap_per_layer"] = avg_spectral_gap

    return {
        "per_layer": per_layer,
        "cumulative": cumulative,
        "summary": summary,
    }


def print_summary(results: dict) -> None:
    """Print human-readable summary of analysis results."""
    summary = results["summary"]
    per_layer = results["per_layer"]
    cumulative = results["cumulative"]

    print("\n" + "=" * 60)
    print("MARKOV MIXING ANALYSIS SUMMARY")
    print("=" * 60)

    print(f"\nModel Configuration:")
    print(f"  Layers (HC modules): {summary['num_layers']}")
    print(f"  Streams: {summary['num_streams']}")

    print(f"\nPer-Layer Statistics (averaged):")
    print(f"  |λ₂| (second eigenvalue): {summary['avg_lambda2_per_layer']:.4f}")
    print(f"  Spectral gap (1 - |λ₂|): {summary['avg_spectral_gap_per_layer']:.4f}")

    print(f"\nCumulative Product Analysis:")
    print(f"  Initial dist to uniform: {summary['initial_dist_uniform']:.4f}")
    print(f"  Final dist to uniform: {summary['final_dist_uniform']:.4f}")
    print(f"  Mixing ratio (final/initial): {summary['mixing_ratio']:.4f}")
    print(f"  Max spectral gap (cumulative): {summary['max_spectral_gap']:.4f}")

    # Interpretation
    print("\n" + "-" * 60)
    print("INTERPRETATION:")

    if summary["avg_lambda2_per_layer"] > 0.9:
        print("  ✓ Per-layer |λ₂| > 0.9: Matrices are near-permutations")
        print("    Markov mixing is likely NOT a problem.")
    elif summary["avg_lambda2_per_layer"] > 0.7:
        print("  ⚠ Per-layer |λ₂| in [0.7, 0.9]: Moderate mixing")
        print("    Some mixing occurs but may not accumulate severely.")
    else:
        print("  ✗ Per-layer |λ₂| < 0.7: Fast mixing")
        print("    Markov mixing is likely causing homogenization.")

    if summary["mixing_ratio"] < 0.5:
        print("  ✗ Mixing ratio < 0.5: Cumulative product converging to uniform")
    elif summary["mixing_ratio"] < 0.8:
        print("  ⚠ Mixing ratio in [0.5, 0.8]: Moderate convergence observed")
    else:
        print("  ✓ Mixing ratio > 0.8: Staying far from uniform")

    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Markov mixing in mHC checkpoints"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to checkpoint file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for JSON results (optional)",
    )
    parser.add_argument(
        "--projection",
        type=str,
        default="sinkhorn",
        choices=["sinkhorn", "orthostochastic"],
        help="Projection method for H_res",
    )
    parser.add_argument(
        "--sinkhorn-iters", type=int, default=10, help="Sinkhorn iterations"
    )
    parser.add_argument(
        "--sinkhorn-tau", type=float, default=0.05, help="Sinkhorn temperature"
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress detailed output"
    )
    args = parser.parse_args()

    # Run analysis
    print(f"Analyzing checkpoint: {args.checkpoint}")
    print(f"Projection method: {args.projection}")

    results = run_analysis(
        args.checkpoint,
        projection=args.projection,
        sinkhorn_iters=args.sinkhorn_iters,
        sinkhorn_tau=args.sinkhorn_tau,
    )

    if "error" in results:
        print(f"Error: {results['error']}")
        return 1

    # Print summary
    if not args.quiet:
        print_summary(results)

    # Save to JSON if output directory specified
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(exist_ok=True, parents=True)

        # Convert tensors to lists for JSON serialization
        json_results = {
            "per_layer": {
                str(k): v for k, v in results["per_layer"].items()
            },
            "cumulative": results["cumulative"],
            "summary": results["summary"],
        }

        output_path = output_dir / "mixing_analysis.json"
        with open(output_path, "w") as f:
            json.dump(json_results, f, indent=2)
        print(f"Results saved to: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

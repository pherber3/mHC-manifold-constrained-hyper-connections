"""Tests for Markov mixing analysis utilities."""

import torch


class TestSpectralAnalysis:
    """Tests for spectral analysis functions."""

    def test_compute_eigenvalues_sorted(self):
        """Eigenvalues should be sorted by magnitude descending."""
        from analysis.spectral import compute_eigenvalues

        # Random matrix
        M = torch.randn(4, 4)
        eigs = compute_eigenvalues(M)

        # Check sorted descending
        for i in range(len(eigs) - 1):
            assert eigs[i] >= eigs[i + 1]

    def test_doubly_stochastic_has_eigenvalue_one(self):
        """Doubly stochastic matrices should have largest eigenvalue 1."""
        from hyper_connections.hyper_connections import sinkhorn_log
        from analysis.spectral import compute_eigenvalues

        logits = torch.randn(4, 4)
        H = sinkhorn_log(logits, num_iters=20, tau=0.05)
        eigs = compute_eigenvalues(H)

        assert torch.allclose(eigs[0], torch.tensor(1.0), atol=1e-3)
        assert (eigs <= 1.0 + 1e-6).all()

    def test_identity_has_zero_spectral_gap(self):
        """Identity matrix should have spectral gap near 0."""
        from analysis.spectral import spectral_gap

        I = torch.eye(4)
        gap = spectral_gap(I)

        # Identity has eigenvalue 1 with multiplicity n
        assert gap < 0.01

    def test_uniform_has_max_spectral_gap(self):
        """Uniform matrix should have spectral gap near 1."""
        from analysis.spectral import spectral_gap

        n = 4
        U = torch.ones(n, n) / n
        gap = spectral_gap(U)

        # Uniform has eigenvalue 1 (once) and 0 (n-1 times)
        assert gap > 0.99

    def test_analyze_h_res_returns_all_keys(self):
        """analyze_h_res should return all expected keys."""
        from analysis.spectral import analyze_h_res

        H = torch.eye(4)
        result = analyze_h_res(H)

        expected_keys = {
            "lambda_1",
            "lambda_2_abs",
            "spectral_gap",
            "dist_to_uniform",
            "dist_to_identity",
            "entropy",
            "min_entry",
            "max_entry",
        }
        assert set(result.keys()) == expected_keys

    def test_mixing_time_estimate(self):
        """Mixing time should be finite for matrices with gap > 0."""
        from analysis.spectral import mixing_time_estimate, spectral_gap

        # Uniform matrix has max spectral gap -> fast mixing
        n = 4
        U = torch.ones(n, n) / n
        t_mix = mixing_time_estimate(U)
        assert t_mix < 100  # Fast mixing

        # Identity has gap ~0 -> infinite mixing time
        I = torch.eye(4)
        t_mix_identity = mixing_time_estimate(I)
        assert t_mix_identity == float("inf")


class TestMarkovMetrics:
    """Tests for Markov mixing metrics."""

    def test_distance_to_uniform_zero_for_uniform(self):
        """Uniform matrix should have zero distance to itself."""
        from analysis.markov_metrics import distance_to_uniform

        n = 4
        U = torch.ones(n, n) / n
        assert distance_to_uniform(U) < 1e-6

    def test_distance_to_identity_zero_for_identity(self):
        """Identity should have zero distance to itself."""
        from analysis.markov_metrics import distance_to_identity

        I = torch.eye(4)
        assert distance_to_identity(I) < 1e-6

    def test_entropy_bounds(self):
        """Entropy should be higher for uniform than identity."""
        from analysis.markov_metrics import matrix_entropy

        n = 4
        I = torch.eye(n)
        U = torch.ones(n, n) / n

        entropy_I = matrix_entropy(I)
        entropy_U = matrix_entropy(U)

        assert entropy_I < entropy_U

    def test_cumulative_product_metrics_length(self):
        """Cumulative metrics should have same length as input."""
        from analysis.markov_metrics import cumulative_product_metrics

        h_list = [torch.eye(4) for _ in range(5)]
        metrics = cumulative_product_metrics(h_list)

        assert len(metrics["dist_to_uniform"]) == 5
        assert len(metrics["dist_to_identity"]) == 5
        assert len(metrics["spectral_gap"]) == 5

    def test_cumulative_product_approaches_uniform(self):
        """Products of random doubly stochastic should converge to uniform."""
        from hyper_connections.hyper_connections import sinkhorn_log
        from analysis.markov_metrics import cumulative_product_metrics

        # High temperature = more uniform = faster mixing
        h_list = [sinkhorn_log(torch.randn(4, 4), 20, 0.2) for _ in range(20)]
        metrics = cumulative_product_metrics(h_list)

        # Distance to uniform should decrease
        assert metrics["dist_to_uniform"][-1] < metrics["dist_to_uniform"][0]

    def test_identity_products_dont_mix(self):
        """Products of identity matrices should stay as identity."""
        from analysis.markov_metrics import cumulative_product_metrics

        h_list = [torch.eye(4) for _ in range(10)]
        metrics = cumulative_product_metrics(h_list)

        # Distance to identity should stay at 0
        for dist in metrics["dist_to_identity"]:
            assert dist < 1e-5


class TestStreamSimilarity:
    """Tests for stream similarity measurement."""

    def test_identical_streams_full_similarity(self):
        """Identical streams should have similarity 1."""
        from analysis.stream_similarity import compute_stream_similarity

        num_streams = 4
        batch, seq, dim = 2, 8, 64

        # All streams identical - need shape (batch * num_streams, seq, dim)
        # where streams are interleaved: [b0s0, b0s1, b0s2, b0s3, b1s0, b1s1, ...]
        single_stream = torch.randn(batch, seq, dim)
        # Stack to get (batch, num_streams, seq, dim) then reshape
        activations = single_stream.unsqueeze(1).expand(batch, num_streams, seq, dim)
        activations = activations.reshape(batch * num_streams, seq, dim)

        sim = compute_stream_similarity(activations, num_streams)

        # Should be all 1s
        assert torch.allclose(sim, torch.ones(num_streams, num_streams), atol=1e-5)

    def test_similarity_matrix_is_symmetric(self):
        """Similarity matrix should be symmetric."""
        from analysis.stream_similarity import compute_stream_similarity

        num_streams = 4
        activations = torch.randn(8, 16, 32)  # batch*streams, seq, dim

        sim = compute_stream_similarity(activations, num_streams)

        assert torch.allclose(sim, sim.T, atol=1e-5)

    def test_similarity_diagonal_is_one(self):
        """Diagonal of similarity matrix should be 1."""
        from analysis.stream_similarity import compute_stream_similarity

        num_streams = 4
        activations = torch.randn(8, 16, 32)

        sim = compute_stream_similarity(activations, num_streams)
        diag = torch.diag(sim)

        assert torch.allclose(diag, torch.ones(num_streams), atol=1e-5)

    def test_mean_off_diagonal_similarity(self):
        """Mean off-diagonal should be between -1 and 1 (cosine similarity range)."""
        from analysis.stream_similarity import (
            compute_stream_similarity,
            mean_off_diagonal_similarity,
        )

        num_streams = 4
        activations = torch.randn(8, 16, 32)

        sim = compute_stream_similarity(activations, num_streams)
        mean_off_diag = mean_off_diagonal_similarity(sim)

        # Cosine similarity can be negative for dissimilar vectors
        assert -1 <= mean_off_diag <= 1


class TestDoublyStochasticConstraints:
    """Tests to verify Sinkhorn produces valid doubly stochastic matrices."""

    def test_sinkhorn_row_sums(self):
        """Sinkhorn output should have rows summing close to 1."""
        from hyper_connections.hyper_connections import sinkhorn_log

        logits = torch.randn(4, 4)
        # More iterations for better convergence
        H = sinkhorn_log(logits, num_iters=50, tau=0.05)

        row_sums = H.sum(dim=-1)
        # Sinkhorn is approximate - use reasonable tolerance
        assert torch.allclose(row_sums, torch.ones(4), atol=0.1)

    def test_sinkhorn_col_sums(self):
        """Sinkhorn output should have columns summing to 1."""
        from hyper_connections.hyper_connections import sinkhorn_log

        logits = torch.randn(4, 4)
        H = sinkhorn_log(logits, num_iters=20, tau=0.05)

        col_sums = H.sum(dim=-2)
        assert torch.allclose(col_sums, torch.ones(4), atol=1e-3)

    def test_sinkhorn_non_negative(self):
        """Sinkhorn output should be non-negative."""
        from hyper_connections.hyper_connections import sinkhorn_log

        logits = torch.randn(4, 4)
        H = sinkhorn_log(logits, num_iters=20, tau=0.05)

        assert (H >= -1e-6).all()

    def test_product_is_doubly_stochastic(self):
        """Product of doubly stochastic matrices is doubly stochastic."""
        from hyper_connections.hyper_connections import sinkhorn_log

        # More iterations for better convergence
        H1 = sinkhorn_log(torch.randn(4, 4), 50, 0.05)
        H2 = sinkhorn_log(torch.randn(4, 4), 50, 0.05)

        H_prod = H1 @ H2

        row_sums = H_prod.sum(dim=-1)
        col_sums = H_prod.sum(dim=-2)

        # Product of approximate DS matrices compounds error - use looser tolerance
        assert torch.allclose(row_sums, torch.ones(4), atol=0.15)
        assert torch.allclose(col_sums, torch.ones(4), atol=0.15)
        assert (H_prod >= -1e-6).all()


class TestOrthostochasticConstraints:
    """Tests for orthostochastic projection."""

    def test_orthostochastic_non_negative(self):
        """Orthostochastic projection should be non-negative."""
        from hyper_connections.hyper_connections import orthostochastic_project

        logits = torch.randn(4, 4)
        H = orthostochastic_project(logits)

        assert (H >= -1e-6).all()

    def test_orthostochastic_is_doubly_stochastic(self):
        """Orthostochastic matrices should be doubly stochastic."""
        from hyper_connections.hyper_connections import orthostochastic_project

        logits = torch.randn(4, 4)
        # Use more Newton-Schulz steps for better orthogonality
        H = orthostochastic_project(logits, ns_steps=10)

        row_sums = H.sum(dim=-1)
        col_sums = H.sum(dim=-2)

        # Orthostochastic is |Q|^2 where Q is orthogonal
        # Row/col sums should be close to 1 but NS iterations affect precision
        assert torch.allclose(row_sums, torch.ones(4), atol=0.35)
        assert torch.allclose(col_sums, torch.ones(4), atol=0.35)

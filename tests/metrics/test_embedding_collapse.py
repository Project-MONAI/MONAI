# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import unittest

import torch

from monai.metrics.embedding_collapse import (
    EmbeddingCollapseMetric,
    _centroid_similarity,
    _domain_shift,
    _effective_rank_score,
    _per_class_rank,
    compute_embedding_collapse,
    linear_probe_accuracy,
)


class TestEmbeddingCollapseMetricInit(unittest.TestCase):
    def test_valid_reductions(self):
        for r in ("max", "mean", "none"):
            m = EmbeddingCollapseMetric(reduction=r)
            self.assertEqual(m.reduction, r)

    def test_invalid_reduction_raises(self):
        with self.assertRaises(ValueError):
            EmbeddingCollapseMetric(reduction="sum")

    def test_include_indicators_stored(self):
        m = EmbeddingCollapseMetric(include_indicators=["centroid_similarity"])
        self.assertEqual(m.include_indicators, ["centroid_similarity"])

    def test_include_indicators_none_by_default(self):
        m = EmbeddingCollapseMetric()
        self.assertIsNone(m.include_indicators)


class TestComputeEmbeddingCollapseReturnTypes(unittest.TestCase):
    """All returned values must be torch.Tensor scalars or None."""

    def setUp(self):
        torch.manual_seed(0)
        self.emb = torch.randn(20, 64)
        self.lbl = torch.randint(0, 2, (20,))

    def test_all_values_are_tensor_or_none(self):
        scores = compute_embedding_collapse(self.emb, self.lbl)
        for k, v in scores.items():
            self.assertTrue(
                v is None or isinstance(v, torch.Tensor),
                f"Key '{k}' has type {type(v).__name__}, expected Tensor or None",
            )

    def test_tensor_values_are_scalar(self):
        scores = compute_embedding_collapse(self.emb, self.lbl)
        for k, v in scores.items():
            if isinstance(v, torch.Tensor):
                self.assertEqual(v.ndim, 0, f"Key '{k}' should be a scalar tensor, got shape {v.shape}")

    def test_tensor_values_in_unit_interval(self):
        scores = compute_embedding_collapse(self.emb, self.lbl)
        for k, v in scores.items():
            if isinstance(v, torch.Tensor):
                self.assertGreaterEqual(float(v), 0.0, f"Key '{k}' = {float(v):.4f} < 0")
                self.assertLessEqual(float(v), 1.0, f"Key '{k}' = {float(v):.4f} > 1")


class TestAggregateReduction(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        self.emb = torch.randn(20, 32)
        self.lbl = torch.randint(0, 2, (20,))

    def test_reduction_max_present(self):
        scores = compute_embedding_collapse(self.emb, self.lbl, reduction="max")
        self.assertIn("aggregate", scores)
        self.assertIsInstance(scores["aggregate"], torch.Tensor)

    def test_reduction_mean_present(self):
        scores = compute_embedding_collapse(self.emb, self.lbl, reduction="mean")
        self.assertIn("aggregate", scores)

    def test_reduction_none_absent(self):
        scores = compute_embedding_collapse(self.emb, self.lbl, reduction="none")
        self.assertNotIn("aggregate", scores)

    def test_reduction_max_geq_mean(self):
        scores_max = compute_embedding_collapse(self.emb, self.lbl, reduction="max")
        scores_mean = compute_embedding_collapse(self.emb, self.lbl, reduction="mean")
        self.assertGreaterEqual(float(scores_max["aggregate"]), float(scores_mean["aggregate"]))


class TestNoLabels(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2)
        self.emb = torch.randn(20, 32)

    def test_centroid_similarity_none_without_labels(self):
        scores = compute_embedding_collapse(self.emb)
        self.assertIsNone(scores["centroid_similarity"])

    def test_separation_none_without_labels(self):
        scores = compute_embedding_collapse(self.emb)
        self.assertIsNone(scores["separation"])

    def test_effective_rank_present_without_labels(self):
        scores = compute_embedding_collapse(self.emb)
        self.assertIsNotNone(scores["effective_rank_score"])

    def test_aggregate_uses_effective_rank_only(self):
        scores = compute_embedding_collapse(self.emb, reduction="max")
        self.assertAlmostEqual(float(scores["aggregate"]), float(scores["effective_rank_score"]), places=5)


class TestCentroidSimilarity(unittest.TestCase):
    def test_identical_centroids_give_one(self):
        # All embeddings identical -> both centroids identical -> score = 1.0
        emb = torch.ones(20, 16)
        lbl = torch.tensor([0] * 10 + [1] * 10, dtype=torch.long)
        score = _centroid_similarity(emb, lbl)
        self.assertIsNotNone(score)
        self.assertAlmostEqual(float(score), 1.0, places=4)

    def test_orthogonal_centroids_give_zero(self):
        # Class 0 centroid = e1, class 1 centroid = e2 -> cosine = 0 -> score = 0.5
        emb = torch.zeros(4, 4)
        emb[0, 0] = 1.0
        emb[1, 0] = 1.0
        emb[2, 1] = 1.0
        emb[3, 1] = 1.0
        lbl = torch.tensor([0, 0, 1, 1], dtype=torch.long)
        score = _centroid_similarity(emb, lbl)
        self.assertIsNotNone(score)
        self.assertAlmostEqual(float(score), 0.5, places=4)

    def test_single_class_returns_none(self):
        emb = torch.randn(10, 8)
        lbl = torch.zeros(10, dtype=torch.long)
        score = _centroid_similarity(emb, lbl)
        self.assertIsNone(score)

    def test_score_in_unit_interval(self):
        torch.manual_seed(3)
        emb = torch.randn(30, 64)
        lbl = torch.randint(0, 3, (30,))
        score = _centroid_similarity(emb, lbl)
        if score is not None:
            self.assertGreaterEqual(float(score), 0.0)
            self.assertLessEqual(float(score), 1.0)

    def test_formula_matches_manual(self):
        torch.manual_seed(4)
        n = 10
        emb_a = torch.randn(n, 16)
        emb_b = torch.randn(n, 16)
        emb = torch.cat([emb_a, emb_b])
        lbl = torch.tensor([0] * n + [1] * n, dtype=torch.long)

        c0 = emb_a.mean(0)
        c1 = emb_b.mean(0)
        c0 = c0 / c0.norm()
        c1 = c1 / c1.norm()
        expected = ((torch.dot(c0, c1) + 1.0) / 2.0).clamp(0.0, 1.0)

        score = _centroid_similarity(emb, lbl)
        self.assertAlmostEqual(float(score), float(expected), places=5)


class TestEffectiveRankScore(unittest.TestCase):
    def test_zero_variance_embeddings_give_one(self):
        """All identical embeddings after centering -> sv_sum == 0 -> score = 1.0 (maximal collapse)."""
        emb = torch.ones(10, 8)  # all identical, centered = all zeros
        score = _effective_rank_score(emb)
        self.assertIsInstance(score, torch.Tensor)
        self.assertAlmostEqual(float(score), 1.0, places=5)

    def test_rank_one_matrix_gives_high_score(self):
        # Near-rank-1: one dominant direction + tiny noise.
        # After mean-centering, the dominant direction survives.
        torch.manual_seed(99)
        n, d = 20, 16
        # Random unit vector as the dominant direction
        direction = torch.randn(1, d)
        direction = direction / direction.norm()
        # All rows point in the same direction with small perturbation
        scales = torch.randn(n, 1) * 5.0  # varying magnitudes
        emb = scales * direction + torch.randn(n, d) * 0.01
        score = _effective_rank_score(emb)
        self.assertGreater(float(score), 0.8)

    def test_random_matrix_lower_score(self):
        torch.manual_seed(5)
        emb = torch.randn(50, 64)
        score = _effective_rank_score(emb)
        # A full-rank random matrix should have low collapse score
        self.assertLess(float(score), 0.8)

    def test_score_in_unit_interval(self):
        torch.manual_seed(6)
        for _ in range(10):
            n = torch.randint(5, 50, (1,)).item()
            d = torch.randint(4, 32, (1,)).item()
            emb = torch.randn(int(n), int(d))
            score = _effective_rank_score(emb)
            self.assertGreaterEqual(float(score), 0.0)
            self.assertLessEqual(float(score), 1.0)

    def test_formula_matches_manual(self):
        torch.manual_seed(7)
        emb = torch.randn(20, 16)
        centered = emb - emb.mean(0)
        _, sv, _ = torch.linalg.svd(centered, full_matrices=False)
        sv_sum = sv.sum()
        probs = sv / sv_sum
        safe_probs = probs.clamp_min(torch.finfo(probs.dtype).tiny)
        eff_rank = (-(probs * safe_probs.log()).sum()).exp()
        expected = (1.0 - eff_rank / min(20, 16)).clamp(0.0, 1.0)
        score = _effective_rank_score(emb)
        self.assertAlmostEqual(float(score), float(expected), places=5)


class TestPerClassRank(unittest.TestCase):
    def test_keys_present_for_each_class(self):
        torch.manual_seed(8)
        emb = torch.randn(20, 16)
        lbl = torch.tensor([0] * 10 + [1] * 10, dtype=torch.long)
        result = _per_class_rank(emb, lbl)
        self.assertIn("per_class_rank_0", result)
        self.assertIn("per_class_rank_1", result)

    def test_single_sample_class_returns_none(self):
        emb = torch.randn(5, 8)
        lbl = torch.tensor([0, 0, 0, 0, 1], dtype=torch.long)
        result = _per_class_rank(emb, lbl)
        self.assertIsNone(result["per_class_rank_1"])

    def test_asymmetric_collapse_detectable(self):
        # Class 0: random full-rank embeddings
        # Class 1: near-rank-1 (one dominant direction + tiny noise)
        torch.manual_seed(9)
        emb_0 = torch.randn(10, 16)

        direction = torch.randn(1, 16)
        direction = direction / direction.norm()
        scales = torch.randn(10, 1) * 5.0
        emb_1 = scales * direction + torch.randn(10, 16) * 0.01

        emb = torch.cat([emb_0, emb_1])
        lbl = torch.tensor([0] * 10 + [1] * 10, dtype=torch.long)
        result = _per_class_rank(emb, lbl)
        score_0 = float(result["per_class_rank_0"])
        score_1 = float(result["per_class_rank_1"])
        # Class 1 (near-rank-1) should be more collapsed than class 0 (random)
        self.assertGreater(score_1, score_0)


class TestDomainShift(unittest.TestCase):
    def test_identical_matrices_give_one(self):
        torch.manual_seed(10)
        src = torch.randn(10, 8)
        score = _domain_shift(src, src.clone())
        self.assertIsNotNone(score)
        self.assertAlmostEqual(float(score), 1.0, places=4)

    def test_score_in_unit_interval(self):
        torch.manual_seed(11)
        src = torch.randn(10, 8)
        tgt = torch.randn(10, 8)
        score = _domain_shift(src, tgt)
        self.assertIsNotNone(score)
        self.assertGreaterEqual(float(score), 0.0)
        self.assertLessEqual(float(score), 1.0)

    def test_single_sample_returns_none(self):
        src = torch.randn(1, 8)
        tgt = torch.randn(10, 8)
        self.assertIsNone(_domain_shift(src, tgt))
        self.assertIsNone(_domain_shift(tgt, src))

    def test_single_sample_target_returns_none_not_raises(self):
        """target_embeddings with 1 sample should return None, not raise ValueError."""
        torch.manual_seed(13)
        emb = torch.randn(10, 8)
        lbl = torch.randint(0, 2, (10,))
        single_target = torch.randn(1, 8)
        # Should not raise — _domain_shift handles n<2 gracefully
        scores = compute_embedding_collapse(emb, lbl, target_embeddings=single_target)
        self.assertIsNone(scores["domain_shift"])

    def test_3d_target_embeddings_raises(self):
        """target_embeddings with wrong ndim should still raise ValueError."""
        emb = torch.randn(10, 8)
        bad_target = torch.randn(5, 8, 2)
        with self.assertRaises(ValueError):
            compute_embedding_collapse(emb, target_embeddings=bad_target)
        torch.manual_seed(12)
        src = torch.randn(20, 8)
        tgt = torch.randn(10, 8)
        score = _domain_shift(src, tgt)
        self.assertIsNotNone(score)
        self.assertGreaterEqual(float(score), 0.0)
        self.assertLessEqual(float(score), 1.0)


class TestValidation(unittest.TestCase):
    def test_1d_input_raises(self):
        with self.assertRaises(ValueError):
            compute_embedding_collapse(torch.randn(10))

    def test_single_sample_raises(self):
        with self.assertRaises(ValueError):
            compute_embedding_collapse(torch.randn(1, 8))

    def test_3d_input_raises(self):
        with self.assertRaises(ValueError):
            compute_embedding_collapse(torch.randn(4, 8, 8))

    def test_invalid_reduction_raises(self):
        with self.assertRaises(ValueError):
            compute_embedding_collapse(torch.randn(10, 8), reduction="sum")

    def test_unknown_include_indicator_raises(self):
        with self.assertRaises(ValueError):
            compute_embedding_collapse(torch.randn(10, 8), include_indicators=["typo_metric"])

    def test_mismatched_labels_shape_raises(self):
        emb = torch.randn(10, 8)
        bad_labels = torch.zeros(5, dtype=torch.long)  # wrong length
        with self.assertRaises(ValueError):
            compute_embedding_collapse(emb, labels=bad_labels)

    def test_2d_labels_raises(self):
        emb = torch.randn(10, 8)
        bad_labels = torch.zeros(10, 2, dtype=torch.long)  # 2D
        with self.assertRaises(ValueError):
            compute_embedding_collapse(emb, labels=bad_labels)


class TestIncludeIndicators(unittest.TestCase):
    def test_only_requested_indicators_computed(self):
        torch.manual_seed(13)
        emb = torch.randn(20, 16)
        lbl = torch.randint(0, 2, (20,))
        scores = compute_embedding_collapse(emb, lbl, include_indicators=["centroid_similarity"])
        self.assertIn("centroid_similarity", scores)
        self.assertNotIn("effective_rank_score", scores)
        self.assertNotIn("separation", scores)

    def test_effective_rank_only(self):
        torch.manual_seed(14)
        emb = torch.randn(20, 16)
        scores = compute_embedding_collapse(emb, include_indicators=["effective_rank"])
        self.assertIn("effective_rank_score", scores)
        self.assertNotIn("centroid_similarity", scores)


class TestCollapsedEmbeddingsDetected(unittest.TestCase):
    """End-to-end: a deliberately collapsed embedding space scores high."""

    def test_collapsed_mlp_scores_above_threshold(self):
        # Simulate a model whose final layer always outputs the same vector
        # regardless of input — perfect collapse.
        n = 40
        emb = torch.ones(n, 32)  # all identical
        lbl = torch.tensor([0] * (n // 2) + [1] * (n // 2), dtype=torch.long)

        scores = compute_embedding_collapse(emb, lbl, reduction="max")

        # centroid_similarity should be 1.0 (identical centroids)
        self.assertAlmostEqual(float(scores["centroid_similarity"]), 1.0, places=4)
        # aggregate should be critical
        self.assertGreater(float(scores["aggregate"]), 0.8)

    def test_healthy_embeddings_score_low(self):
        # Well-separated classes: class 0 = +e1, class 1 = -e1
        torch.manual_seed(15)
        n = 20
        emb = torch.zeros(n, 16)
        emb[: n // 2, 0] = 1.0  # class 0 centroid = +e1
        emb[n // 2 :, 0] = -1.0  # class 1 centroid = -e1
        lbl = torch.tensor([0] * (n // 2) + [1] * (n // 2), dtype=torch.long)

        scores = compute_embedding_collapse(emb, lbl, reduction="max")

        # centroid_similarity should be 0.0 (opposite centroids -> raw cosine = -1 -> score = 0)
        self.assertAlmostEqual(float(scores["centroid_similarity"]), 0.0, places=4)


class TestLinearProbeAccuracy(unittest.TestCase):
    def test_linearly_separable_data_high_accuracy(self):
        # Class 0: positive first dim, Class 1: negative first dim
        n = 50
        train_emb = torch.zeros(n, 8)
        train_emb[: n // 2, 0] = 1.0
        train_emb[n // 2 :, 0] = -1.0
        train_lbl = torch.tensor([0] * (n // 2) + [1] * (n // 2), dtype=torch.long)

        test_emb = train_emb.clone()
        test_lbl = train_lbl.clone()

        try:
            acc = linear_probe_accuracy(train_emb, train_lbl, test_emb, test_lbl)
            self.assertIsInstance(acc, torch.Tensor)
            self.assertGreater(float(acc), 0.9)
        except ImportError:
            self.skipTest("scikit-learn not installed")

    def test_returns_tensor(self):
        torch.manual_seed(16)
        emb = torch.randn(20, 8)
        lbl = torch.randint(0, 2, (20,))
        try:
            acc = linear_probe_accuracy(emb, lbl, emb, lbl)
            self.assertIsInstance(acc, torch.Tensor)
            self.assertEqual(acc.ndim, 0)
            self.assertGreaterEqual(float(acc), 0.0)
            self.assertLessEqual(float(acc), 1.0)
        except ImportError:
            self.skipTest("scikit-learn not installed")

    def test_raises_without_sklearn(self):
        from unittest.mock import patch

        with patch("monai.metrics.embedding_collapse.has_sklearn_linear", False):
            with self.assertRaises(ImportError):
                linear_probe_accuracy(
                    torch.randn(10, 4),
                    torch.zeros(10, dtype=torch.long),
                    torch.randn(5, 4),
                    torch.zeros(5, dtype=torch.long),
                )


if __name__ == "__main__":
    unittest.main()

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

import warnings
from collections.abc import Sequence

import torch

from monai.metrics.metric import Metric
from monai.utils import optional_import

sklearn_silhouette, has_sklearn = optional_import("sklearn.metrics", name="silhouette_score")
sklearn_logistic, has_sklearn_linear = optional_import("sklearn.linear_model", name="LogisticRegression")

__all__ = ["EmbeddingCollapseMetric", "compute_embedding_collapse"]

_VALID_REDUCTIONS = ("max", "mean", "none")
_VALID_INDICATORS = frozenset({"centroid_similarity", "effective_rank", "per_class_rank", "domain_shift", "separation"})


class EmbeddingCollapseMetric(Metric):
    """Measures representational collapse in neural network embedding spaces.

    Representational collapse occurs when a model's internal embeddings lose
    discriminative power - class centroids converge, the effective dimensionality
    of the embedding space collapses, or source and target domains become
    indistinguishable in representation space. This can happen silently: task
    metrics like AUROC or Dice remain unchanged while the embedding space
    becomes degenerate.

    This metric computes a suite of collapse indicators from a batch of
    embeddings and optional class labels. All scores are in **[0, 1]** where
    **higher = more collapsed**.

    Follows the ``FIDMetric`` / ``MMDMetric`` architectural pattern:
    tensor-in, tensor-out, no I/O side effects. Core dependencies are
    ``torch`` only. ``scikit-learn`` is imported lazily via
    ``optional_import`` and is only required for ``separation`` and
    ``linear_probe_accuracy``.

    Args:
        reduction: how to aggregate individual scores into ``aggregate``.
            ``"max"`` returns the worst-case score (recommended for
            safety-critical use). ``"mean"`` returns the average of
            available scores. ``"none"`` omits the ``aggregate`` key.
        include_indicators: optional list of indicator names to compute.
            If ``None``, all applicable indicators are computed.
            Valid names: ``"centroid_similarity"``, ``"effective_rank"``,
            ``"per_class_rank"``, ``"domain_shift"``, ``"separation"``.

    References:
        - Roy, O. & Vetterli, M. (2007). The effective rank: A measure of
          effective dimensionality. *EUSIPCO*.
        - Kornblith, S. et al. (2019). Similarity of neural network
          representations revisited. *ICML*.
        - Hua, T. et al. (2021). On feature decorrelation in self-supervised
          learning. *ICCV*.

    Examples:
        >>> metric = EmbeddingCollapseMetric()
        >>> emb = torch.randn(100, 768)
        >>> labels = torch.randint(0, 2, (100,))
        >>> scores = metric(embeddings=emb, labels=labels)
        >>> scores["centroid_similarity"]   # tensor scalar in [0, 1]
        >>> scores["effective_rank_score"]  # tensor scalar in [0, 1]
        >>> scores["aggregate"]             # worst-case score
    """

    def __init__(self, reduction: str = "max", include_indicators: Sequence[str] | None = None) -> None:
        super().__init__()
        if reduction not in _VALID_REDUCTIONS:
            raise ValueError(f"reduction must be one of {_VALID_REDUCTIONS}, got '{reduction}'")
        if include_indicators is not None:
            unknown = set(include_indicators) - _VALID_INDICATORS
            if unknown:
                raise ValueError(f"Unknown include_indicators: {unknown}. Valid: {_VALID_INDICATORS}")
        self.reduction = reduction
        self.include_indicators = list(include_indicators) if include_indicators is not None else None

    def __call__(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor | None = None,
        target_embeddings: torch.Tensor | None = None,
        target_labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        """Compute collapse scores.

        Args:
            embeddings: float tensor of shape ``[N, D]``. Required.
            labels: integer class labels of shape ``[N]``. Required for
                ``centroid_similarity``, ``per_class_rank``, and
                ``separation``. If ``None``, only label-free indicators
                are computed.
            target_embeddings: optional second embedding matrix ``[M, D]``
                for cross-domain ``domain_shift`` (CKA) computation.
            target_labels: unused; reserved for future class-conditional
                domain shift computation.

        Returns:
            Dictionary mapping indicator name to a scalar ``torch.Tensor``
            in ``[0, 1]``, or ``None`` when the indicator is not applicable.
            Always includes ``"aggregate"`` unless ``reduction="none"``.

        Raises:
            ValueError: if ``embeddings`` has fewer than 2 samples, is not
                2-D, or ``labels`` shape does not match ``embeddings``.
        """
        return compute_embedding_collapse(
            embeddings=embeddings,
            labels=labels,
            target_embeddings=target_embeddings,
            target_labels=target_labels,
            reduction=self.reduction,
            include_indicators=self.include_indicators,
        )


def compute_embedding_collapse(
    embeddings: torch.Tensor,
    labels: torch.Tensor | None = None,
    target_embeddings: torch.Tensor | None = None,
    target_labels: torch.Tensor | None = None,
    reduction: str = "max",
    include_indicators: Sequence[str] | None = None,
) -> dict[str, torch.Tensor | None]:
    """Functional form of :class:`EmbeddingCollapseMetric`.

    Computes a suite of representational collapse indicators from embeddings
    and optional class labels. All scores are in **[0, 1]** where higher
    values indicate more severe collapse.

    Args:
        embeddings: float tensor of shape ``[N, D]``.
        labels: integer class labels of shape ``[N]``, or ``None``.
        target_embeddings: optional second embedding matrix ``[M, D]``
            for cross-domain CKA computation.
        target_labels: unused; reserved for future use.
        reduction: ``"max"``, ``"mean"``, or ``"none"``.
        include_indicators: subset of indicators to compute, or ``None``
            for all applicable indicators.

    Returns:
        Dictionary with scalar tensor values or ``None``:

        - ``centroid_similarity``: cosine similarity between L2-normalised
          class centroids. ``None`` if fewer than 2 classes.
        - ``effective_rank_score``: dimensional collapse score via SVD
          effective rank. Always present.
        - ``per_class_rank_<cls>``: per-class effective rank score.
        - ``domain_shift``: linear CKA between source and target domains.
          ``None`` if ``target_embeddings`` not provided.
        - ``separation``: silhouette-based inter-class separation score.
          ``None`` if sklearn unavailable or fewer than 2 classes.
        - ``aggregate``: reduced score. Omitted when ``reduction="none"``.

    Raises:
        ValueError: if inputs are invalid (shape, reduction, indicators).
    """
    _validate_embeddings(embeddings)
    emb = embeddings.float()

    if reduction not in _VALID_REDUCTIONS:
        raise ValueError(f"reduction must be one of {_VALID_REDUCTIONS}, got '{reduction}'")

    if include_indicators is not None:
        unknown = set(include_indicators) - _VALID_INDICATORS
        if unknown:
            raise ValueError(f"Unknown include_indicators: {unknown}. Valid: {_VALID_INDICATORS}")

    inc = set(include_indicators) if include_indicators is not None else None
    scores: dict[str, torch.Tensor | None] = {}

    # -- Label-dependent indicators
    if labels is not None:
        lbl = labels.long()
        if lbl.ndim != 1 or lbl.shape[0] != emb.shape[0]:
            raise ValueError(f"labels must be 1-D with shape [{emb.shape[0]}], got {tuple(lbl.shape)}")

        if inc is None or "centroid_similarity" in inc:
            scores["centroid_similarity"] = _centroid_similarity(emb, lbl)

        if inc is None or "per_class_rank" in inc:
            scores.update(_per_class_rank(emb, lbl))

        if inc is None or "separation" in inc:
            scores["separation"] = _separation(emb, lbl)
    else:
        if inc is None or "centroid_similarity" in inc:
            scores["centroid_similarity"] = None
        if inc is None or "separation" in inc:
            scores["separation"] = None

    # -- Label-free indicators
    if inc is None or "effective_rank" in inc:
        scores["effective_rank_score"] = _effective_rank_score(emb)

    # -- Cross-domain indicator
    if inc is None or "domain_shift" in inc:
        if target_embeddings is not None:
            if target_embeddings.ndim != 2:
                raise ValueError(f"target_embeddings must be 2-D [M, D], got shape {tuple(target_embeddings.shape)}")
            scores["domain_shift"] = _domain_shift(emb, target_embeddings.float())
        else:
            scores["domain_shift"] = None

    # -- Aggregate
    if reduction != "none":
        primary = {"centroid_similarity", "effective_rank_score", "domain_shift", "separation"}
        available = [v.to(device=emb.device) for k, v in scores.items() if k in primary and v is not None]
        if not available:
            scores["aggregate"] = None
        elif reduction == "max":
            scores["aggregate"] = torch.stack(available).max()
        else:
            scores["aggregate"] = torch.stack(available).mean()

    return scores


# ---------------------------------------------------------------------------
# Individual indicator functions
# ---------------------------------------------------------------------------


def _centroid_similarity(emb: torch.Tensor, labels: torch.Tensor) -> torch.Tensor | None:
    """Cosine similarity between L2-normalised class centroids.

    Args:
        emb: ``[N, D]`` float tensor.
        labels: ``[N]`` integer class labels.

    Returns:
        Scalar tensor in ``[0, 1]``. ``None`` if fewer than 2 classes.
        1.0 = centroids identical (full collapse).
        0.5 = centroids orthogonal (cosine = 0).
        0.0 = centroids anti-parallel (cosine = -1, maximum separation).
    """
    unique = torch.unique(labels)
    if unique.numel() < 2:
        return None

    centroids = []
    for cls in unique:
        c = emb[labels == cls].mean(dim=0)
        norm = c.norm(p=2)
        centroids.append(c / norm if norm > 0 else c)

    ct = torch.stack(centroids)
    n = ct.shape[0]

    if n == 2:
        raw = torch.dot(ct[0], ct[1])
    else:
        sim = ct @ ct.T
        pairs = torch.stack([sim[i, j] for i in range(n) for j in range(i + 1, n)])
        raw = pairs.mean()

    return ((raw + 1.0) / 2.0).clamp(0.0, 1.0)


def _effective_rank_score(emb: torch.Tensor) -> torch.Tensor:
    """Dimensional collapse score via SVD effective rank.

    Uses the entropy-based effective rank from Roy & Vetterli (2007):
    ``eff_rank = sum(sv) / max(sv)`` where ``sv`` are the singular values
    of the mean-centred embedding matrix.

    If all singular values are zero (constant embeddings after centering),
    the matrix is rank-0 — the most extreme form of dimensional collapse —
    and the function returns ``1.0``.

    Args:
        emb: ``[N, D]`` float tensor.

    Returns:
        Scalar tensor in ``[0, 1]``. 1.0 = full dimensional collapse.
        0.0 = full-rank uniform spectrum (no collapse).
    """
    centered = emb - emb.mean(dim=0, keepdim=True)
    _, sv, _ = torch.linalg.svd(centered, full_matrices=False)
    sv_sum = sv.sum()
    if sv_sum == 0:
        return emb.new_tensor(1.0)
    probs = sv / sv_sum
    safe_probs = probs.clamp_min(torch.finfo(probs.dtype).tiny)
    eff_rank: torch.Tensor = torch.exp(-(probs * safe_probs.log()).sum())
    max_rank: torch.Tensor = emb.new_tensor(float(min(emb.shape[0], emb.shape[1])))
    return (emb.new_tensor(1.0) - eff_rank / max_rank).clamp(0.0, 1.0)


def _per_class_rank(emb: torch.Tensor, labels: torch.Tensor) -> dict[str, torch.Tensor | None]:
    """Effective rank score computed separately per class.

    Detects asymmetric collapse: one class may use 400 dimensions while
    another collapses to 6, which global SVD would average away.

    Args:
        emb: ``[N, D]`` float tensor.
        labels: ``[N]`` integer class labels.

    Returns:
        Dict mapping ``"per_class_rank_<cls>"`` to scalar tensor or ``None``.
    """
    result: dict[str, torch.Tensor | None] = {}
    for cls in torch.unique(labels):
        cls_emb = emb[labels == cls]
        key = f"per_class_rank_{cls.item()}"
        if cls_emb.shape[0] < 2:
            result[key] = None
        else:
            result[key] = _effective_rank_score(cls_emb)
    return result


def _domain_shift(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor | None:
    """Linear CKA between source and target embedding matrices.

    Reference: Kornblith et al. (2019).

    Args:
        source: ``[N, D]`` float tensor.
        target: ``[M, D]`` float tensor.

    Returns:
        Scalar tensor in ``[0, 1]``, or ``None`` if either set has < 2 samples.
        1.0 = representations identical. 0.0 = representations orthogonal.
    """
    if source.shape[0] < 2 or target.shape[0] < 2:
        return None

    if source.shape[0] != target.shape[0]:
        n = min(source.shape[0], target.shape[0])
        g = torch.Generator()
        g.manual_seed(42)
        if source.shape[0] > n:
            source = source[torch.randperm(source.shape[0], generator=g)[:n]]
        else:
            target = target[torch.randperm(target.shape[0], generator=g)[:n]]

    hsic_xy = _hsic(source, target)
    hsic_xx = _hsic(source, source)
    hsic_yy = _hsic(target, target)
    denom = (hsic_xx * hsic_yy).sqrt()
    if denom == 0.0:
        return source.new_tensor(0.0)
    return (hsic_xy / denom).clamp(0.0, 1.0)


def _separation(emb: torch.Tensor, labels: torch.Tensor) -> torch.Tensor | None:
    """Silhouette-based inter-class separation score.

    Requires scikit-learn. Returns ``None`` if unavailable.

    Args:
        emb: ``[N, D]`` float tensor.
        labels: ``[N]`` integer class labels.

    Returns:
        Scalar tensor in ``[0, 1]``. 1.0 = no separation (collapsed).
        0.0 = perfect separation. ``None`` if sklearn unavailable or
        fewer than 2 classes.
    """
    if not has_sklearn:
        warnings.warn(
            "scikit-learn is not installed; 'separation' score skipped. " "Install with: pip install scikit-learn",
            stacklevel=3,
        )
        return None

    unique = torch.unique(labels)
    if unique.numel() < 2:
        return None

    try:
        sil = sklearn_silhouette(  # type: ignore[operator]
            emb.detach().cpu().numpy(), labels.detach().cpu().numpy(), metric="cosine"
        )
        return torch.tensor((1.0 - float(sil)) / 2.0, dtype=emb.dtype, device=emb.device).clamp(0.0, 1.0)
    except Exception as exc:
        warnings.warn(f"separation: silhouette_score failed: {exc}", stacklevel=3)
        return None


def _hsic(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Unbiased linear HSIC estimator."""
    n = x.shape[0]
    gram_x = x @ x.T
    gram_y = y @ y.T
    centering = torch.eye(n, dtype=x.dtype, device=x.device) - torch.ones(n, n, dtype=x.dtype, device=x.device) / n
    return torch.sum((centering @ gram_x @ centering) * (centering @ gram_y @ centering)) / ((n - 1) ** 2)


def _validate_embeddings(embeddings: torch.Tensor) -> None:
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2-D [N, D], got shape {tuple(embeddings.shape)}")
    if embeddings.shape[0] < 2:
        raise ValueError(f"Need at least 2 samples, got {embeddings.shape[0]}.")


# ---------------------------------------------------------------------------
# Utility: linear probe accuracy
# ---------------------------------------------------------------------------


def linear_probe_accuracy(
    train_embeddings: torch.Tensor,
    train_labels: torch.Tensor,
    test_embeddings: torch.Tensor,
    test_labels: torch.Tensor,
    max_iter: int = 1000,
) -> torch.Tensor:
    """Fit a linear classifier on embeddings and return test accuracy.

    Used to validate that collapse scores predict downstream task performance.

    Requires scikit-learn.

    Args:
        train_embeddings: ``[N_train, D]`` float tensor.
        train_labels: ``[N_train]`` integer labels.
        test_embeddings: ``[N_test, D]`` float tensor.
        test_labels: ``[N_test]`` integer labels.
        max_iter: max iterations for ``LogisticRegression``.

    Returns:
        Scalar tensor: test accuracy in ``[0, 1]``.

    Raises:
        ImportError: if scikit-learn is not installed.
    """
    if not has_sklearn_linear:
        raise ImportError(
            "scikit-learn is required for linear_probe_accuracy. " "Install with: pip install scikit-learn"
        )

    clf = sklearn_logistic(max_iter=max_iter, random_state=42)  # type: ignore[operator]
    clf.fit(train_embeddings.detach().float().cpu().numpy(), train_labels.detach().cpu().numpy())
    preds = clf.predict(test_embeddings.detach().float().cpu().numpy())
    acc = (preds == test_labels.detach().cpu().numpy()).mean()
    return torch.tensor(float(acc))

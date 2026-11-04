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

"""Conformal risk control for segmentation / classification (issue #8935, part 2).

Implements the recipe of Angelopoulos, Bates, Lei, Wasserman & Jordan,
"Conformal Risk Control" (arXiv:2208.02814, 2022): pick a single threshold
``lambda_hat`` on a held-out calibration split via the finite-sample-corrected
selection ``lambda_hat = inf { lambda : (n * R_hat(lambda) + B) / (n + 1) <= alpha }``
(``B = 1`` is the loss upper bound), which guarantees ``E[L] <= alpha`` on a
fresh sample. At inference, the same threshold yields a prediction set per
voxel / per sample, and the per-voxel *uncertainty mask* flags locations where
the set contains more than one class (i.e. the model is ambiguous).

This module mirrors the :class:`ConformalPredictor` / :class:`ConformalCalibrator`
split from ``monai/inferers/conformal_predictor.py`` but lives in ``metrics``
because the calibration target is a *loss bounded on a calibration set*
rather than a marginal-coverage quantile, and because the natural outputs
(``Coverage``, ``SetSize``) are evaluation metrics.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import torch

from monai.metrics.metric import CumulativeIterationMetric
from monai.utils import MetricReduction
from monai.utils.module import optional_import

__all__ = [
    "ConformalRiskCalibrator",
    "ConformalRiskPredictor",
    "Coverage",
    "SetSize",
    "compute_coverage",
    "compute_set_size",
]

tqdm, has_tqdm = optional_import("tqdm", name="tqdm")


# ----------------------------------------------------------------------------------------------
# Losses for conformal risk control. Each returns a per-sample (image-level) loss in [0, 1]
# given the prediction-set mask (B, C, spatial...) and the integer label (B, 1, spatial...).
# ----------------------------------------------------------------------------------------------


def _set_from_threshold(scores: torch.Tensor, lam: float) -> torch.Tensor:
    """Boolean prediction set ``{ y : score(y) <= lam }``, shape (..., C).

    ``scores`` is the non-conformity score tensor with class as the last dim;
    ``lam`` is a scalar threshold. Returns a bool tensor of the same shape.
    """
    return scores <= lam


def _flatten_spatial(sets: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Bring ``(B, C, spatial...)`` set mask and ``(B, 1, spatial...)``/``(B, spatial...)`` labels
    to ``(N, C)`` and ``(N,)`` so the same loss code serves classification and segmentation."""
    if sets.ndim < 2:
        raise ValueError(f"sets must be (B, C, spatial...), got shape {tuple(sets.shape)}.")
    c = sets.shape[1]
    sets_flat = sets.movedim(1, -1).reshape(-1, c)
    labels_flat = labels.reshape(-1).long()
    if (labels_flat < 0).any() or (labels_flat >= c).any():
        raise ValueError(
            f"labels must lie in [0, {c - 1}], got min={int(labels_flat.min())}, max={int(labels_flat.max())}."
        )
    return sets_flat, labels_flat


def miscoverage_loss(sets: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Per-image mean miscoverage ``mean_voxel 1{ y_v not in S_v }`` in [0, 1].

    This is the canonical loss for conformal risk control on a classification/
    segmentation output (Angelopoulos et al. 2022, Eq. 11). ``sets`` is a bool
    tensor ``(B, C, spatial...)`` and ``labels`` is ``(B, 1, spatial...)`` or
    ``(B, spatial...)`` integer class indices. Returns ``(B,)`` per-image loss.
    """
    sets_flat, labels_flat = _flatten_spatial(sets, labels)
    b = sets.shape[0]
    n_per_image = sets_flat.shape[0] // b
    covered = sets_flat.gather(1, labels_flat.unsqueeze(1)).squeeze(1).bool()
    miss = (~covered).float()
    return miss.reshape(b, n_per_image).mean(dim=1)


def false_negative_loss(sets: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Per-image false-negative rate among foreground voxels, in [0, 1].

    Useful when background dominates and a pure miscoverage target is too lax for
    the classes that matter. Voxels where ``labels == 0`` are excluded from the
    denominator. Returns ``(B,)`` per-image loss; images with no foreground
    voxels get ``0`` (so they do not push ``lambda`` down).
    """
    sets_flat, labels_flat = _flatten_spatial(sets, labels)
    b = sets.shape[0]
    n_per_image = sets_flat.shape[0] // b
    covered = sets_flat.gather(1, labels_flat.unsqueeze(1)).squeeze(1).bool()
    fg = labels_flat != 0
    miss = (~covered).float()
    miss = miss * fg.float()
    miss = miss.reshape(b, n_per_image)
    denom = fg.float().reshape(b, n_per_image).sum(dim=1).clamp(min=1.0)
    return miss.sum(dim=1) / denom


_LOSSES: dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
    "miscoverage": miscoverage_loss,
    "false_negative": false_negative_loss,
}


# ----------------------------------------------------------------------------------------------
# Calibration / prediction
# ----------------------------------------------------------------------------------------------


class ConformalRiskCalibrator:
    """Calibrate a single threshold ``lambda_hat`` that bounds an image-level loss on a
    held-out split, following Conformal Risk Control (Angelopoulos et al. 2022, arXiv:2208.02814).

    Unlike split-conformal (which targets marginal coverage via a quantile), risk control
    picks ``lambda_hat = inf { lambda : (n * R_hat(lambda) + B) / (n + 1) <= alpha }`` where
    ``R_hat(lambda) = (1/n) sum_i L(y_i, S_lambda(x_i))`` and ``B = 1`` bounds the loss; this
    guarantees ``E[L(Y, S_lambda_hat(X))] <= alpha`` on a fresh sample. The threshold is
    global — one scalar applied to every voxel / sample at inference. When ``alpha`` is too
    small for the calibration size (``alpha < 1 / (n + 1)``) no threshold satisfies the bound
    and :meth:`calibrate` falls back to the largest grid value (full sets).

    The non-conformity score is ``1 - softmax[y]`` (LAC, same as
    :class:`monai.inferers.ConformalCalibrator`); the set at threshold ``lambda`` is
    ``S_lambda(x) = { y : 1 - softmax[y] <= lambda }``.

    Args:
        alpha: target risk, e.g. ``0.1`` bounds the expected loss at ~``0.1``.
        loss: image-level loss bounded in [0, 1]. Either a callable
            ``(sets, labels) -> (B,)`` tensor, or one of ``"miscoverage"`` /
            ``"false_negative"``. A callable **must be non-increasing in the threshold**
            ``lambda`` (larger sets never increase the loss) — this is a precondition of
            the CRC guarantee (Angelopoulos et al. 2022, Thm 1); both built-ins satisfy
            it. :meth:`calibrate` warns if the empirical risk is not non-increasing over
            the grid.
        include_background: when ``False`` drop background-labeled (class 0) voxels from
            the score pool before computing the loss. Defaults to ``True``.
        lam_grid: grid of candidate thresholds (1-D tensor in ``[0, 1]``) used for the
            ``inf`` search. Defaults to ``torch.linspace(0, 1, 101)``. Finer grids give a
            tighter bound at a small compute cost.

    Example:

        .. code-block:: python

            import torch
            from monai.metrics import ConformalRiskCalibrator

            cal = ConformalRiskCalibrator(alpha=0.1, loss="miscoverage")
            for batch in cal_loader:
                probs = model(batch["image"]).softmax(dim=1)
                cal.accumulate(probs, batch["label"])
            lam = cal.calibrate()
            # lam is a scalar threshold; pass it to ConformalRiskPredictor

    References:
        - Angelopoulos, A.; Bates, S.; Lei, J.; Wasserman, L.; Jordan, M. "Conformal
          Risk Control." arXiv:2208.02814, 2022. https://arxiv.org/abs/2208.02814
    """

    def __init__(
        self,
        alpha: float = 0.1,
        loss: str | Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = "miscoverage",
        include_background: bool = True,
        lam_grid: torch.Tensor | None = None,
    ) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
        if isinstance(loss, str):
            if loss not in _LOSSES:
                raise ValueError(f"Unknown loss {loss!r}; available: {sorted(_LOSSES)}.")
            loss_fn = _LOSSES[loss]
        elif callable(loss):
            loss_fn = loss
        else:
            raise TypeError(f"loss must be a str or callable, got {type(loss)}.")
        self.alpha = float(alpha)
        self.loss_fn = loss_fn
        self.include_background = include_background
        if lam_grid is None:
            lam_grid = torch.linspace(0.0, 1.0, 101)
        if lam_grid.ndim != 1 or lam_grid.numel() == 0 or (lam_grid < 0).any() or (lam_grid > 1).any():
            raise ValueError("lam_grid must be a non-empty 1-D tensor with values in [0, 1].")
        if not bool((lam_grid[1:] >= lam_grid[:-1]).all()):
            raise ValueError("lam_grid must be sorted in ascending order for the infimum search.")
        self.lam_grid = lam_grid.float()
        # Per-image score/label tensors, stored one entry per calibration image so spatial
        # size may vary across images and across accumulate() calls (variable-size volumes).
        self._scores: list[torch.Tensor] = []  # each (P_i, C)
        self._labels: list[torch.Tensor] = []  # each (P_i,)
        self._num_classes: int | None = None

    def accumulate(self, probs: torch.Tensor, labels: torch.Tensor) -> None:
        """Accumulate calibration data from one batch.

        Spatial size may differ from batch to batch; each image is stored separately so
        calibration works on variable-size volumes. The channel count ``C`` must stay fixed.

        Args:
            probs: softmax probabilities ``(B, C, spatial...)`` in [0, 1] summing to 1 over C.
            labels: integer class indices ``(B, 1, spatial...)`` or ``(B, spatial...)`` in [0, C).
        """
        if probs.ndim < 2:
            raise ValueError(f"probs must be (B, C, spatial...), got shape {tuple(probs.shape)}.")
        b, c = probs.shape[:2]
        if self._num_classes is None:
            self._num_classes = c
        elif c != self._num_classes:
            raise ValueError(f"channel count C changed across accumulate() calls: {self._num_classes} -> {c}.")
        spatial = probs.shape[2:]
        per_image = int(torch.tensor(spatial).prod().item()) if spatial else 1
        # (B, per_image, C): move class to last then flatten spatial
        scores = (1.0 - probs).movedim(1, -1).reshape(b, per_image, c).detach()
        # labels (B, 1, spatial...) or (B, spatial...) -> (B, per_image)
        labels_flat = labels.reshape(b, per_image).long().detach()
        if (labels_flat < 0).any() or (labels_flat >= c).any():
            raise ValueError(
                f"labels must lie in [0, {c - 1}], got min={int(labels_flat.min())}, max={int(labels_flat.max())}."
            )
        for i in range(b):
            self._scores.append(scores[i])  # (per_image, C)
            self._labels.append(labels_flat[i])  # (per_image,)

    def calibrate(self) -> torch.Tensor:
        """Search ``lam_grid`` for the smallest threshold whose risk-controlled bound holds.

        Selects ``lambda_hat = inf { lambda : (n * R_hat(lambda) + B) / (n + 1) <= alpha }``
        with ``B = 1`` (Angelopoulos et al. 2022, Thm 1), which bounds the expected loss on a
        fresh sample by ``alpha``.

        Returns:
            Scalar tensor ``lambda_hat``. If no grid point satisfies the finite-sample bound
            (only possible when ``alpha < 1 / (n + 1)`` — the calibration set is too small for
            the requested risk), the largest grid value is returned (full sets); callers
            should check the achieved risk with :class:`Coverage` / :class:`SetSize`.
        """
        if not self._scores:
            raise RuntimeError("No calibration data accumulated; call accumulate(probs, labels) first.")
        n = len(self._scores)
        device, dtype = self._scores[0].device, self._scores[0].dtype
        lam_grid = self.lam_grid.to(device)
        n_lam = lam_grid.numel()
        # Sum each image's per-lambda loss; images vary in size so we loop per image but
        # vectorize over the whole lambda grid (n_lam acts as the batch dim into loss_fn).
        risk_sum = torch.zeros(n_lam, device=device, dtype=torch.float32)
        # Chunk over the lambda grid to bound peak memory; the full
        # (n_lam, P_i, C) tensor would OOM on large 3D volumes. 1 << 12 lambdas
        # at a time keeps the working set modest while preserving the cumulative
        # sum; lower if calibration volumes are very large.
        lam_chunk = 1 << 12
        for scores_i, labels_i in zip(self._scores, self._labels, strict=True):
            if not self.include_background:
                keep = labels_i != 0
                if not bool(keep.any()):
                    continue  # all-background image: 0 loss, but still counted in n
                scores_i, labels_i = scores_i[keep], labels_i[keep]
            p_i = scores_i.shape[0]
            for start in range(0, n_lam, lam_chunk):
                end = min(start + lam_chunk, n_lam)
                lam_chunk_grid = lam_grid[start:end]  # (n_chunk,)
                sets = scores_i.unsqueeze(0) <= lam_chunk_grid.view(-1, 1, 1)  # (n_chunk, P_i, C)
                sets_shaped = sets.movedim(-1, 1)  # (n_chunk, C, P_i)
                labels_rep = labels_i.view(1, 1, -1).expand(sets_shaped.shape[0], 1, p_i)  # (n_chunk, 1, P_i)
                loss = self.loss_fn(sets_shaped, labels_rep).float()
                if loss.shape != (sets_shaped.shape[0],):
                    raise ValueError(
                        f"loss_fn must return per-image loss of shape (n_chunk,), got {tuple(loss.shape)}."
                    )
                if bool(torch.isnan(loss).any()):
                    raise ValueError("loss_fn returned NaN; check inputs or loss implementation.")
                risk_sum[start:end] += loss
        emp_risk = risk_sum / n
        # CRC requires the loss to be non-increasing in lambda; a violating custom loss
        # breaks the infimum selection and voids the E[L] <= alpha guarantee.
        if not bool((emp_risk[1:] <= emp_risk[:-1] + 1e-6).all()):
            warnings.warn(
                "empirical risk is not non-increasing in lambda; the conformal risk control "
                "guarantee requires a loss that is non-increasing in the threshold. "
                "Check the custom loss function.",
                stacklevel=2,
            )
        # Finite-sample-corrected selection. B = 1 is the loss upper bound (losses are in
        # [0, 1]); losses are non-increasing in lambda, so the leftmost lambda clearing the
        # bound is the infimum.
        b_bound = 1.0
        alpha_eff = ((n + 1) * self.alpha - b_bound) / n
        within = (emp_risk <= alpha_eff).nonzero(as_tuple=True)[0]
        if within.numel() == 0:
            lam_hat = lam_grid[-1]
        else:
            lam_hat = lam_grid[within[0]]
        self.reset()  # one-shot; caller keeps lam_hat
        return lam_hat.to(dtype).to(device)

    def reset(self) -> None:
        """Reset internal calibration state.

        Clears the per-image score/label buffers and the cached class count so
        the calibrator can be reused on a fresh calibration split.
        """
        self._scores, self._labels = [], []
        self._num_classes = None


class ConformalRiskPredictor:
    """Apply a pre-calibrated threshold ``lambda_hat`` at inference and return both the
    prediction set and the per-voxel *uncertainty mask*.

    The uncertainty mask flags voxels where the prediction set contains more than one
    class — i.e. the model cannot commit to a single label at the calibrated risk level.
    Voxels in the mask are candidates for review, defer-to-human, or downstream refinement.

    This is intentionally *not* an :class:`monai.inferers.Inferer` subclass: it does not
    own a network. Pair it with any inferer (e.g. ``SimpleInferer`` or
    ``SlidingWindowInferer``) that produces logits, then call this on the softmax.

    Args:
        lam: calibrated threshold (scalar tensor). Required.
        include_background: if ``False``, background voxels (label 0 at inference = argmax
            0) are excluded from the uncertainty mask. Defaults to ``True``.

    Example:

        .. code-block:: python

            import torch
            from monai.inferers import SlidingWindowInferer
            from monai.metrics import ConformalRiskPredictor

            lam = torch.tensor(0.4)
            crp = ConformalRiskPredictor(lam=lam)
            with torch.no_grad():
                logits = sliding_inferer(imgs, model)
            sets, mask, probs = crp(logits.softmax(dim=1))
            # sets: (B, C, ...) bool, mask: (B, 1, ...) bool, probs: (B, C, ...) float
    """

    def __init__(self, lam: torch.Tensor, include_background: bool = True) -> None:
        self.set_threshold(lam)
        self.include_background = include_background

    def set_threshold(self, lam: torch.Tensor) -> None:
        """Set (or update) the calibrated threshold.

        Args:
            lam: scalar tensor in ``[0, 1]``. A non-scalar would broadcast over
                spatial dims at inference and silently produce wrong sets.
        """
        if not isinstance(lam, torch.Tensor):
            raise TypeError(f"lam must be a torch.Tensor, got {type(lam)}.")
        if lam.ndim != 0:
            raise ValueError(f"lam must be a scalar tensor, got shape {tuple(lam.shape)}.")
        lam_val = float(lam.detach().item())
        if not 0.0 <= lam_val <= 1.0:
            raise ValueError(f"lam must lie in [0, 1], got {lam_val}.")
        self.lam = lam.detach().clone()

    def __call__(self, probs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run inference-time risk-controlled prediction.

        Args:
            probs: softmax probabilities ``(B, C, spatial...)`` in [0, 1] summing to 1 over C.

        Returns:
            A 3-tuple ``(sets, uncertainty_mask, probs)``. ``sets`` is a bool tensor
            ``(B, C, spatial...)`` with ``True`` where class ``c`` is in the set.
            ``uncertainty_mask`` is a bool tensor ``(B, 1, spatial...)`` with ``True`` where the
            set holds more than one class (ambiguous voxels), zeroed at background-argmax voxels
            when ``include_background=False``. ``probs`` is the input, returned for convenience
            (e.g. for :class:`Coverage` / :class:`SetSize`).
        """
        if probs.ndim < 2:
            raise ValueError(f"probs must be (B, C, spatial...), got shape {tuple(probs.shape)}.")
        lam = self.lam.to(probs.device, probs.dtype)
        sets = (1.0 - probs) <= lam
        # per-voxel set size > 1 -> ambiguous
        set_size = sets.sum(dim=1, keepdim=True)
        uncertainty_mask = set_size > 1
        if not self.include_background:
            # background voxels are where argmax == 0; zero them out of the mask
            argmax = probs.argmax(dim=1, keepdim=True)
            uncertainty_mask = uncertainty_mask & (argmax != 0)
        return sets, uncertainty_mask, probs


# ----------------------------------------------------------------------------------------------
# Evaluation metrics for the prediction sets. These are CumulativeIterationMetrics so they
# compose with MONAI's evaluator / handler infrastructure like DiceMetric.
# ----------------------------------------------------------------------------------------------


def compute_coverage(sets: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Per-image fraction of voxels whose true label is in the prediction set, in [0, 1].

    Args:
        sets: bool tensor ``(B, C, spatial...)``.
        labels: integer class indices ``(B, 1, spatial...)`` or ``(B, spatial...)`` in [0, C).

    Returns:
        ``(B,)`` per-image coverage. Higher is better (1 = full coverage).
    """
    sets_flat, labels_flat = _flatten_spatial(sets, labels)
    b = sets.shape[0]
    n_per_image = sets_flat.shape[0] // b
    covered = sets_flat.gather(1, labels_flat.unsqueeze(1)).squeeze(1).bool()
    return covered.float().reshape(b, n_per_image).mean(dim=1)


def compute_set_size(sets: torch.Tensor) -> torch.Tensor:
    """Per-image mean prediction-set size (number of classes in the set per voxel), in [0, C].

    Args:
        sets: bool tensor ``(B, C, spatial...)``.

    Returns:
        ``(B,)`` per-image mean set size. Smaller is better (tight sets).
    """
    b = sets.shape[0]
    sizes = sets.sum(dim=1).float()  # (B, spatial...)
    return sizes.reshape(b, -1).mean(dim=1)


class Coverage(CumulativeIterationMetric):
    """Cumulative per-image coverage of conformal prediction sets.

    Coverage = fraction of voxels / samples whose true label is inside the prediction set.
    For a well-calibrated split-conformal or risk-controlled predictor this should be
    ``>= 1 - alpha`` on a held-out test set (split-conformal) or satisfy the risk bound
    (risk control). Useful as a sanity check after calibration.

    Args:
        metric_reduction: reduction across batch/channel dims on ``aggregate()``.
            Defaults to ``"mean"``.
        get_not_nans: if ``True``, ``aggregate()`` returns ``(metric, not_nans)``.

    Example:

        .. code-block:: python

            import torch
            from monai.metrics import Coverage, ConformalRiskPredictor

            cov = Coverage()
            predictor = ConformalRiskPredictor(lam=torch.tensor(0.4))
            for batch in test_loader:
                probs = model(batch["image"]).softmax(dim=1)
                sets, _, _ = predictor(probs)
                cov(sets, batch["label"])
            print(cov.aggregate())
    """

    def __init__(
        self, metric_reduction: MetricReduction | str = MetricReduction.MEAN, get_not_nans: bool = False
    ) -> None:
        super().__init__()
        self.metric_reduction = metric_reduction
        self.get_not_nans = get_not_nans

    def _compute_tensor(self, y_pred: torch.Tensor, y: torch.Tensor, **kwargs: Any) -> torch.Tensor:  # type: ignore[override]
        if not isinstance(y_pred, torch.Tensor) or not isinstance(y, torch.Tensor):
            raise TypeError("Coverage expects torch.Tensor inputs (sets, labels).")
        return compute_coverage(y_pred, y).unsqueeze(1)  # (B, 1) for do_metric_reduction

    def aggregate(
        self, reduction: MetricReduction | str | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        from monai.metrics.utils import do_metric_reduction

        data = self.get_buffer()
        if not isinstance(data, torch.Tensor):
            raise ValueError("the data to aggregate must be a PyTorch Tensor.")
        f, not_nans = do_metric_reduction(data, reduction or self.metric_reduction)
        return (f, not_nans) if self.get_not_nans else f


class SetSize(CumulativeIterationMetric):
    """Cumulative per-image mean prediction-set size.

    Set size = average number of classes in the prediction set per voxel / sample. Smaller
    is better (tighter sets). Use alongside :class:`Coverage` to check the coverage /
    efficiency trade-off of a conformal predictor.

    Args:
        metric_reduction: reduction across batch/channel dims on ``aggregate()``.
            Defaults to ``"mean"``.
        get_not_nans: if ``True``, ``aggregate()`` returns ``(metric, not_nans)``.
    """

    def __init__(
        self, metric_reduction: MetricReduction | str = MetricReduction.MEAN, get_not_nans: bool = False
    ) -> None:
        super().__init__()
        self.metric_reduction = metric_reduction
        self.get_not_nans = get_not_nans

    def _compute_tensor(self, y_pred: torch.Tensor, y: torch.Tensor | None = None, **kwargs: Any) -> torch.Tensor:  # type: ignore[override]
        if not isinstance(y_pred, torch.Tensor):
            raise TypeError("SetSize expects a torch.Tensor input (sets).")
        return compute_set_size(y_pred).unsqueeze(1)

    def aggregate(
        self, reduction: MetricReduction | str | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        from monai.metrics.utils import do_metric_reduction

        data = self.get_buffer()
        if not isinstance(data, torch.Tensor):
            raise ValueError("the data to aggregate must be a PyTorch Tensor.")
        f, not_nans = do_metric_reduction(data, reduction or self.metric_reduction)
        return (f, not_nans) if self.get_not_nans else f

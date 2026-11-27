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

import math
from collections.abc import Callable
from typing import Any

import torch

from monai.inferers.inferer import Inferer
from monai.utils.module import optional_import

__all__ = ["ConformalPredictor", "ConformalCalibrator"]

tqdm, has_tqdm = optional_import("tqdm", name="tqdm")


def _quantile_threshold(scores: torch.Tensor, alpha: float) -> torch.Tensor:
    """Split-conformal threshold ``qhat`` at the ceil((n+1)(1-alpha))/n quantile of ``scores``.

    Implements the finite-sample marginal coverage guarantee of split conformal prediction
    (Vovk et al. 2005; Angelopoulos & Bates 2021). ``scores`` is a 1-D tensor of
    non-conformity scores from the held-out calibration split; the returned scalar
    lives on the same device/dtype as ``scores``.

    Args:
        scores: 1-D tensor of non-conformity scores from the calibration split.
        alpha: mis-coverage level in ``(0, 1)``.

    Returns:
        Scalar tensor ``qhat`` on the same device/dtype as ``scores``. When
        ``ceil((n+1)(1-alpha)) > n`` (calibration set too small for the requested ``alpha``),
        returns ``+inf`` so every class is admitted, preserving the coverage guarantee.

    Raises:
        ValueError: if ``scores`` is empty or ``alpha`` is not in ``(0, 1)``.
    """
    n = scores.numel()
    if n <= 0:
        raise ValueError("Cannot calibrate from an empty calibration set.")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
    rank = math.ceil((n + 1) * (1.0 - alpha))
    if rank > n:
        # too few calibration points for this alpha: no finite score gives the guarantee,
        # so return +inf (full prediction set) rather than under-covering at the max score.
        return torch.tensor(float("inf"), dtype=scores.dtype, device=scores.device)
    rank = max(1, rank)  # kthvalue is 1-indexed
    return torch.kthvalue(scores.float(), rank).values.to(scores.dtype)


class ConformalCalibrator:
    """Collect softmax on a held-out calibration split and turn it into a split-conformal
    threshold using the LAC (Least Ambiguous set-Valued Classifier) non-conformity score
    ``s_i = 1 - p_i(y_i)`` (Sadinle et al. 2019, arXiv:1905.12581).

    The calibrator is decoupled from any network/transform pipeline: it consumes softmax
    probabilities and integer labels, so it works for image-level classification directly
    and for per-voxel segmentation by reshaping ``(B, C, spatial...)`` to ``(N, C)``.

    Args:
        alpha: mis-coverage level, e.g. ``0.1`` gives 90% marginal coverage.
        score: non-conformity score, currently only ``"lac"`` (``1 - softmax[y]``).
        include_background: when ``False`` exclude background-labeled (class 0) voxels from the
            calibration set (useful for segmentation where background voxels dominate).

    Example:

        .. code-block:: python

            import torch
            from monai.inferers.conformal_predictor import ConformalCalibrator

            cal = ConformalCalibrator(alpha=0.1)
            for batch in cal_loader:
                logits = model(batch["image"])          # (B, C, ...)
                probs  = logits.softmax(dim=1)
                cal.accumulate(probs, batch["label"])   # label: (B, 1, ...) int
            qhat = cal.calibrate()
            # qhat is the threshold used by ConformalPredictor

    """

    def __init__(self, alpha: float = 0.1, score: str = "lac", include_background: bool = True) -> None:
        if score != "lac":
            raise ValueError(f"Unsupported score {score!r}; only 'lac' (1 - softmax[y]) is implemented.")
        self.alpha = float(alpha)
        self.score = score
        self.include_background = include_background
        self._scores: list[torch.Tensor] = []

    def accumulate(self, probs: torch.Tensor, labels: torch.Tensor) -> None:
        """Accumulate non-conformity scores from one calibration batch.

        Args:
            probs: softmax probabilities ``(B, C, spatial...)`` in ``[0, 1]`` summing to 1 over C.
            labels: integer class indices ``(B, 1, spatial...)`` or ``(B, spatial...)`` with values
                in ``[0, C)``. Shape must broadcast against ``probs`` spatial dims.

        Raises:
            ValueError: if ``probs`` has fewer than 2 dimensions.
        """
        if probs.ndim < 2:
            raise ValueError(f"probs must be (B, C, spatial...), got shape {tuple(probs.shape)}.")
        c = probs.shape[1]
        # flatten to (N, C)
        probs_flat = probs.reshape(probs.shape[0], c, -1).movedim(1, -1).reshape(-1, c)
        labels_flat = labels.reshape(-1).long()
        if not self.include_background:
            # drop background-labeled voxels (class 0) from calibration so the threshold isn't
            # dominated by easy background; the full softmax is kept so ``1 - softmax[y]`` stays a
            # valid LAC score. (Relabeling/renormalizing instead would corrupt the scores.)
            keep = labels_flat != 0
            probs_flat = probs_flat[keep]
            labels_flat = labels_flat[keep]
        # reject invalid labels (negative or >= C) outright rather than silently clamping them,
        # which would corrupt the non-conformity scores.
        valid = (labels_flat >= 0) & (labels_flat < c)
        probs_flat = probs_flat[valid]
        labels_flat = labels_flat[valid]
        if labels_flat.numel() == 0:
            return  # nothing to accumulate this batch (all labels invalid or all bg-excluded)
        true_p = probs_flat.gather(1, labels_flat.unsqueeze(1)).squeeze(1)
        # move to CPU to avoid GPU OOM on large per-voxel calibration sets
        self._scores.append((1.0 - true_p).detach().cpu())

    def calibrate(self) -> torch.Tensor:
        """Return the split-conformal threshold ``qhat`` from all accumulated scores.

        Returns:
            Scalar tensor ``qhat`` on CPU.

        Raises:
            RuntimeError: if no calibration scores have been accumulated.
        """
        if not self._scores:
            raise RuntimeError("No calibration scores accumulated; call accumulate(probs, labels) first.")
        all_scores = torch.cat(self._scores)
        qhat = _quantile_threshold(all_scores, self.alpha)
        self._scores = []  # one-shot; caller keeps qhat
        return qhat

    def reset(self) -> None:
        self._scores = []


class ConformalPredictor(Inferer):
    """Inferer that wraps a network and a pre-calibrated split-conformal threshold ``qhat``
    to return prediction sets with a marginal coverage guarantee ``1 - alpha``.

    This implements the LAC (Least Ambiguous set-Valued Classifier) recipe from issue #8935:

        1. (out of band) calibrate ``qhat`` on a held-out split with
           :class:`ConformalCalibrator` using non-conformity score ``1 - softmax[y]``.
        2. at inference compute ``softmax(logits)`` and return, for each sample, the set
           ``{ y : 1 - softmax[y] <= qhat }``.

    Args:
        qhat: pre-calibrated threshold (scalar tensor). Pass ``None`` to use ``alpha`` together
            with calibration data passed via :meth:`calibrate`; otherwise ``qhat`` is used directly.
        alpha: mis-coverage level used only by :meth:`calibrate` when ``qhat`` is ``None``.
        score: non-conformity score, currently only ``"lac"``.
        include_background: forwarded to :class:`ConformalCalibrator` for :meth:`calibrate`.

    The inferer calls ``network(inputs, *args, **kwargs)`` exactly as :class:`SimpleInferer`
    does, then applies ``softmax`` over channel dim 1 and builds the set.

    References:
        - Sadinle, M.; Lei, J.; Wasserman, L. "Least Ambiguous Set-Valued Classifiers with
          Bounded Error Rates." arXiv:1905.12581, 2019. https://arxiv.org/abs/1905.12581
        - Angelopoulos, A.; Bates, S. "A Gentle Introduction to Conformal Prediction and
          Distribution-Free Uncertainty Quantification." arXiv:2107.07511, 2021.

    Example:

        .. code-block:: python

            import torch
            from monai.inferers import ConformalPredictor

            qhat = torch.tensor(0.65)
            inferer = ConformalPredictor(qhat=qhat)
            with torch.no_grad():
                sets = inferer(imgs, model)        # sets: (B, C, ...) bool
            # sets[b, c, ...] True means class c is in the prediction set at that location.

    """

    def __init__(
        self, qhat: torch.Tensor | None = None, alpha: float = 0.1, score: str = "lac", include_background: bool = True
    ) -> None:
        Inferer.__init__(self)
        self.score = score
        self.alpha = float(alpha)
        self.include_background = include_background
        self.qhat: torch.Tensor | None = None
        if qhat is not None:
            self.set_threshold(qhat)

    def set_threshold(self, qhat: torch.Tensor) -> None:
        """Set (or update) the calibrated threshold. Lets you keep one inferer and re-calibrate.

        Args:
            qhat: scalar ``torch.Tensor`` threshold.

        Raises:
            TypeError: if ``qhat`` is not a ``torch.Tensor``.
            ValueError: if ``qhat`` is not a scalar (has more than one element).
        """
        if not isinstance(qhat, torch.Tensor):
            raise TypeError(f"qhat must be a torch.Tensor, got {type(qhat)}.")
        if qhat.numel() != 1:
            raise ValueError(f"qhat must be a scalar tensor, got {qhat.numel()} elements.")
        self.qhat = qhat.detach().clone()

    def calibrate(
        self, network: torch.nn.Module, cal_loader: Any, device: torch.device | str | None = None
    ) -> torch.Tensor:
        """Run the network on ``cal_loader`` to calibrate ``qhat`` in-band.

        Args:
            network: ``nn.Module`` returning logits ``(B, C, spatial...)`` (needs ``.parameters()``
                for device inference and ``.eval()``).
            cal_loader: iterable yielding ``dict``-like items with keys ``"image"`` and ``"label"``.
                ``"label"`` is integer class indices ``(B, 1, ...)`` or ``(B, ...)``.
            device: device to run the network on; defaults to ``next(network.parameters()).device``.

        Returns:
            The calibrated ``qhat`` (also stored and used by subsequent ``__call__`` invocations).

        Raises:
            TypeError: if the network returns a non-Tensor.
        """
        if device is None:
            param = next(network.parameters(), None)
            device = param.device if param is not None else torch.device("cpu")
        was_training = getattr(network, "training", False)
        network.eval()
        cal = ConformalCalibrator(alpha=self.alpha, score=self.score, include_background=self.include_background)
        iterator = cal_loader
        if has_tqdm:
            iterator = tqdm(cal_loader, desc="conformal calibration")
        with torch.no_grad():
            for batch in iterator:
                imgs = batch["image"]
                if isinstance(imgs, torch.Tensor):
                    imgs = imgs.to(device)
                logits = network(imgs)
                if isinstance(logits, torch.Tensor):
                    probs = logits.softmax(dim=1).detach()
                else:
                    raise TypeError(f"network must return a Tensor of logits, got {type(logits)}.")
                cal.accumulate(probs.to(device), batch["label"].to(device))
        qhat = cal.calibrate()
        self.set_threshold(qhat)
        if was_training:
            network.train()
        return qhat

    def __call__(
        self, inputs: torch.Tensor, network: Callable[..., torch.Tensor], *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        """Run inference and return the prediction-set mask.

        Args:
            inputs: input batch ``(B, C_in, spatial...)``.
            network: callable returning logits ``(B, C, spatial...)``.
            args/kwargs: forwarded to ``network``.

        Returns:
            sets: bool tensor ``(B, C, spatial...)``, ``True`` where class ``c`` is in the set.
            For the underlying softmax, call ``network(inputs).softmax(1)``.

        Raises:
            RuntimeError: if no threshold has been set.
            TypeError: if the network returns a non-Tensor.
        """
        if self.qhat is None:
            raise RuntimeError("No threshold set; call set_threshold(qhat) or calibrate(...) first.")
        logits = network(inputs, *args, **kwargs)
        if not isinstance(logits, torch.Tensor):
            raise TypeError(f"network must return a Tensor of logits, got {type(logits)}.")
        probs = logits.softmax(dim=1)
        qhat = self.qhat.to(probs.device, probs.dtype)
        sets: torch.Tensor = (1.0 - probs) <= qhat
        return sets

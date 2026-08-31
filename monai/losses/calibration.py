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
from collections.abc import Callable, Sequence

import torch
from torch.nn.modules.loss import _Loss

from monai.networks import one_hot
from monai.utils import LossReduction

__all__ = ["HardL1ACELoss", "SoftL1ACELoss"]


def _accumulation_dtype(input: torch.Tensor) -> torch.dtype:
    return torch.float32 if input.dtype in (torch.float16, torch.bfloat16) else input.dtype


def _hard_binned_calibration(
    input: torch.Tensor, target: torch.Tensor, num_bins: int, right: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return hard-binned prediction sums, target sums, and counts."""
    work_dtype = _accumulation_dtype(input)
    input_flat = input.flatten(start_dim=2).to(dtype=work_dtype).contiguous()
    target_flat = target.detach().flatten(start_dim=2).to(dtype=work_dtype)

    # Match calibration_binning's established boundaries, including its epsilon-expanded upper edge.
    # Spell out float32 epsilon because torch.finfo is not supported by TorchScript.
    float32_eps = 1.1920928955078125e-7
    boundaries = torch.linspace(0.0, 1.0 + float32_eps, num_bins + 1, dtype=work_dtype, device=input.device)
    bin_idx = torch.bucketize(input_flat, boundaries[1:], right=right).clamp(max=num_bins - 1)
    counts = torch.zeros(input_flat.shape[0], input_flat.shape[1], num_bins, dtype=work_dtype, device=input.device)
    counts = counts.scatter_add(2, bin_idx, torch.ones_like(input_flat))
    sum_p = torch.zeros_like(counts).scatter_add(2, bin_idx, input_flat)
    sum_target = torch.zeros_like(counts).scatter_add(2, bin_idx, target_flat)
    return sum_p, sum_target, counts


def _soft_binned_calibration(
    input: torch.Tensor, target: torch.Tensor, num_bins: int, right: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return soft-binned prediction sums, target sums, and effective counts."""
    work_dtype = _accumulation_dtype(input)
    input_flat = input.flatten(start_dim=2).to(dtype=work_dtype).contiguous()
    target_flat = target.detach().flatten(start_dim=2).to(dtype=work_dtype)

    # Spell out float32 epsilon because torch.finfo is not supported by TorchScript.
    float32_eps = 1.1920928955078125e-7
    half_boundaries = torch.linspace(0.0, 1.0 + float32_eps, 2 * num_bins + 1, dtype=work_dtype, device=input.device)
    centers = half_boundaries[1::2].contiguous()
    insertion_idx = torch.bucketize(input_flat, centers, right=right)
    left_idx = (insertion_idx - 1).clamp(min=0, max=num_bins - 1)
    right_idx = insertion_idx.clamp(max=num_bins - 1)

    left_centers = centers[left_idx]
    right_centers = centers[right_idx]
    distinct = left_idx != right_idx
    distance = (right_centers - left_centers).clamp_min(float32_eps)
    right_weight = torch.where(distinct, (input_flat - left_centers) / distance, torch.zeros_like(input_flat))
    left_weight = 1.0 - right_weight

    counts = torch.zeros(input_flat.shape[0], input_flat.shape[1], num_bins, dtype=work_dtype, device=input.device)
    counts = counts.scatter_add(2, left_idx, left_weight).scatter_add(2, right_idx, right_weight)
    sum_p = torch.zeros_like(counts)
    sum_p = sum_p.scatter_add(2, left_idx, left_weight * input_flat)
    sum_p = sum_p.scatter_add(2, right_idx, right_weight * input_flat)
    sum_target = torch.zeros_like(counts)
    sum_target = sum_target.scatter_add(2, left_idx, left_weight * target_flat)
    sum_target = sum_target.scatter_add(2, right_idx, right_weight * target_flat)
    return sum_p, sum_target, counts


class _L1ACELoss(_Loss):
    """Shared input handling and reduction for marginal L1 ACE losses."""

    def __init__(
        self,
        num_bins: int,
        include_background: bool,
        to_onehot_y: bool,
        sigmoid: bool,
        softmax: bool,
        other_act: Callable | None,
        reduction: LossReduction | str,
        weight: Sequence[float] | float | int | torch.Tensor | None,
        right: bool,
        ignore_empty_classes: bool,
    ) -> None:
        super().__init__(reduction=LossReduction(reduction).value)
        if num_bins < 1:
            raise ValueError(f"num_bins must be >= 1, got {num_bins}.")
        if other_act is not None and not callable(other_act):
            raise TypeError(f"other_act must be None or callable but is {type(other_act).__name__}.")
        if int(sigmoid) + int(softmax) + int(other_act is not None) > 1:
            raise ValueError("Incompatible values: more than 1 of [sigmoid=True, softmax=True, other_act is not None].")

        class_weight = torch.as_tensor(weight) if weight is not None else None
        if class_weight is not None:
            if class_weight.ndim > 1:
                raise ValueError("weight must be a scalar or a one-dimensional sequence.")
            if torch.any(class_weight < 0):
                raise ValueError("the value/values of the `weight` should be no less than 0.")

        self.num_bins = num_bins
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.other_act = other_act
        self.right = right
        self.ignore_empty_classes = ignore_empty_classes
        self.register_buffer("class_weight", class_weight)
        self.class_weight: None | torch.Tensor

    def _prepare_input(self, input: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if input.ndim < 3:
            raise ValueError(f"input must have shape (B, C, spatial...), got ndim={input.ndim}.")
        if not input.is_floating_point():
            raise TypeError(f"input must be a floating point tensor, got {input.dtype}.")

        if self.sigmoid:
            input = torch.sigmoid(input)

        n_pred_ch = input.shape[1]
        if self.softmax:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `softmax=True` ignored.", stacklevel=3)
            else:
                input = torch.softmax(input, 1)
        if self.other_act is not None:
            input = self.other_act(input)

        if self.to_onehot_y:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `to_onehot_y=True` ignored.", stacklevel=3)
            else:
                target = one_hot(target, num_classes=n_pred_ch)

        if not self.include_background:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `include_background=False` ignored.", stacklevel=3)
            else:
                input = input[:, 1:]
                target = target[:, 1:]

        if target.shape != input.shape:
            raise AssertionError(f"ground truth has different shape ({target.shape}) from input ({input.shape})")
        return input, target

    def _reduce(self, per_class_loss: torch.Tensor, valid_classes: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
        num_classes = per_class_loss.shape[1]
        if self.class_weight is not None:
            if self.class_weight.ndim == 0:
                class_weight = self.class_weight.expand(num_classes)
            elif self.class_weight.shape[0] == num_classes:
                class_weight = self.class_weight
            else:
                raise ValueError(
                    "The length of the `weight` sequence should be the same as the number of classes. "
                    "If `include_background=False`, the weight should not include the background category class 0."
                )
            per_class_loss = per_class_loss * class_weight.to(per_class_loss)

        per_class_loss = per_class_loss * valid_classes.to(dtype=per_class_loss.dtype)
        if self.reduction == LossReduction.MEAN.value:
            return per_class_loss.mean()
        if self.reduction == LossReduction.SUM.value:
            return per_class_loss.sum()
        if self.reduction == LossReduction.NONE.value:
            broadcast_shape = list(per_class_loss.shape) + [1] * (input.ndim - 2)
            return per_class_loss.view(broadcast_shape)
        raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')

    def _finish(
        self,
        sum_p: torch.Tensor,
        sum_target: torch.Tensor,
        counts: torch.Tensor,
        valid_bins: torch.Tensor,
        input: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        safe_counts = torch.where(valid_bins, counts, torch.ones_like(counts))
        gap = torch.abs(sum_p / safe_counts - sum_target / safe_counts)
        valid_bins_float = valid_bins.to(dtype=gap.dtype)
        valid_bin_count = valid_bins_float.sum(dim=-1)
        per_class_loss = (gap * valid_bins_float).sum(dim=-1) / valid_bin_count.clamp_min(1)
        valid_classes = valid_bin_count > 0
        if self.ignore_empty_classes:
            valid_classes = valid_classes & (target.flatten(start_dim=2).sum(dim=-1) > 0)
        return self._reduce(per_class_loss, valid_classes, input)


class HardL1ACELoss(_L1ACELoss):
    """
    Compute hard-binned marginal L1 Average Calibration Error (ACE) loss.

    The loss measures calibration independently for every image and class. Predicted probabilities are assigned to
    hard bins, the absolute difference between mean probability and mean binary target is computed in each occupied
    bin, and those differences are averaged equally. Unlike Expected Calibration Error, occupied bins are not
    weighted by voxel count. Hard assignments are discrete, while the mean probability within a fixed assignment is
    differentiable.

    Input must have shape ``(B, C, spatial...)``. It is interpreted as probabilities unless ``sigmoid``, ``softmax``,
    or ``other_act`` is selected. Targets may have the same one-hot shape or be label maps of shape
    ``(B, 1, spatial...)`` when ``to_onehot_y=True``. Ignored empty target classes contribute zero before reduction,
    matching the reference implementation; with ``reduction="none"`` they are returned as zero. Class weights are
    applied before reduction.
    The unreduced shape is ``(B, C, 1, ..., 1)`` after optional background removal.

    Finite-bin calibration estimates depend on the bin count and data distribution. This loss is intended as an
    auxiliary objective and may trade segmentation accuracy against calibration quality; tune its coefficient and
    ``num_bins`` on validation data.

    Args:
        num_bins: number of equally spaced bins. Defaults to 20.
        include_background: whether channel 0 contributes. Defaults to ``True``.
        to_onehot_y: convert a single-channel label map to one-hot targets. Defaults to ``False``.
        sigmoid: apply sigmoid to input.
        softmax: apply channel-wise softmax to input.
        other_act: optional callable activation. Only one activation option may be used.
        reduction: one of ``"none"``, ``"mean"``, or ``"sum"``.
        weight: scalar or one non-negative value per included class.
        right: hard-bin boundary inclusion rule, matching :py:func:`monai.metrics.calibration_binning`.
        ignore_empty_classes: set the loss for classes absent from an image to zero before reduction. Defaults to
            ``True``.

    See Also:
        - :py:func:`monai.metrics.calibration_binning`: The corresponding calibration bin statistics.
        - :py:class:`monai.metrics.CalibrationErrorMetric`: Evaluation metrics computed from those statistics.

    References:
        - Barfoot et al., "Average Calibration Losses for Reliable Uncertainty in Medical Image Segmentation,"
          IEEE Transactions on Medical Imaging, 2026. https://doi.org/10.1109/TMI.2026.3673118
        - Barfoot et al., MICCAI 2024. https://papers.miccai.org/miccai-2024/091-Paper3075.html
        - Guo et al., "On Calibration of Modern Neural Networks," ICML 2017.

    Example:
        >>> import torch
        >>> from monai.losses import DiceCELoss, HardL1ACELoss
        >>> logits, labels = torch.randn(2, 3, 16, 16), torch.randint(0, 3, (2, 1, 16, 16))
        >>> segmentation = DiceCELoss(to_onehot_y=True, softmax=True)
        >>> calibration = HardL1ACELoss(to_onehot_y=True, softmax=True)
        >>> loss = segmentation(logits, labels) + 0.1 * calibration(logits, labels)
    """

    def __init__(
        self,
        num_bins: int = 20,
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = False,
        softmax: bool = False,
        other_act: Callable | None = None,
        reduction: LossReduction | str = LossReduction.MEAN,
        weight: Sequence[float] | float | int | torch.Tensor | None = None,
        right: bool = False,
        ignore_empty_classes: bool = True,
    ) -> None:
        super().__init__(
            num_bins,
            include_background,
            to_onehot_y,
            sigmoid,
            softmax,
            other_act,
            reduction,
            weight,
            right,
            ignore_empty_classes,
        )

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the loss between ``input`` and ``target``."""
        input, target = self._prepare_input(input, target)
        sum_p, sum_target, counts = _hard_binned_calibration(input, target, self.num_bins, self.right)
        return self._finish(sum_p, sum_target, counts, counts > 0, input, target)


class SoftL1ACELoss(_L1ACELoss):
    """
    Compute soft-binned marginal L1 Average Calibration Error (ACE) loss.

    This loss has the same per-image, per-class and occupied-bin averaging semantics as :class:`HardL1ACELoss`, but
    linearly interpolates each probability between its two neighboring bin centers. The implementation stores only
    two indices and weights per probability, keeping memory complexity ``O(B*C*N)`` rather than materializing an
    ``(B, C, N, num_bins)`` tensor. Soft binning provides a smoother training signal across bin transitions.

    Input must have shape ``(B, C, spatial...)`` and contain probabilities unless an activation is enabled. Targets
    may be matching one-hot tensors or single-channel label maps when ``to_onehot_y=True``. Bins whose effective
    weight is below ``empty_weight`` are ignored. Empty target classes can contribute zero before reduction, and
    ``reduction="none"`` returns ``(B, C, 1, ..., 1)`` with ignored entries set to zero.

    Finite-bin estimates and soft assignments depend on ``num_bins``, ``empty_weight``, and the sample distribution.
    Use this loss as a validated auxiliary objective: improved calibration can coincide with lower segmentation
    performance.

    Args:
        num_bins: number of equally spaced bin centers. Defaults to 20.
        include_background: whether channel 0 contributes. Defaults to ``True``.
        to_onehot_y: convert a single-channel label map to one-hot targets. Defaults to ``False``.
        sigmoid: apply sigmoid to input.
        softmax: apply channel-wise softmax to input.
        other_act: optional callable activation. Only one activation option may be used.
        reduction: one of ``"none"``, ``"mean"``, or ``"sum"``.
        weight: scalar or one non-negative value per included class.
        empty_weight: minimum effective bin weight. Empty bins are always ignored. Defaults to 0.01.
        right: boundary rule used when a probability equals a bin center.
        ignore_empty_classes: set the loss for classes absent from an image to zero before reduction. Defaults to
            ``True``.

    See Also:
        - :py:func:`monai.metrics.calibration_binning`: Hard-binned evaluation statistics for reliability diagrams.
        - :py:class:`monai.metrics.CalibrationErrorMetric`: Evaluation metrics computed from those statistics.

    References:
        - Barfoot et al., "Average Calibration Losses for Reliable Uncertainty in Medical Image Segmentation,"
          IEEE Transactions on Medical Imaging, 2026. https://doi.org/10.1109/TMI.2026.3673118
        - Barfoot et al., MICCAI 2024. https://papers.miccai.org/miccai-2024/091-Paper3075.html
        - Guo et al., "On Calibration of Modern Neural Networks," ICML 2017.

    Example:
        >>> import torch
        >>> from monai.losses import DiceCELoss, SoftL1ACELoss
        >>> logits, labels = torch.randn(2, 3, 16, 16), torch.randint(0, 3, (2, 1, 16, 16))
        >>> segmentation = DiceCELoss(to_onehot_y=True, softmax=True)
        >>> calibration = SoftL1ACELoss(to_onehot_y=True, softmax=True)
        >>> loss = segmentation(logits, labels) + 0.1 * calibration(logits, labels)
    """

    def __init__(
        self,
        num_bins: int = 20,
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = False,
        softmax: bool = False,
        other_act: Callable | None = None,
        reduction: LossReduction | str = LossReduction.MEAN,
        weight: Sequence[float] | float | int | torch.Tensor | None = None,
        empty_weight: float = 0.01,
        right: bool = False,
        ignore_empty_classes: bool = True,
    ) -> None:
        if empty_weight < 0:
            raise ValueError(f"empty_weight must be >= 0, got {empty_weight}.")
        super().__init__(
            num_bins,
            include_background,
            to_onehot_y,
            sigmoid,
            softmax,
            other_act,
            reduction,
            weight,
            right,
            ignore_empty_classes,
        )
        self.empty_weight = float(empty_weight)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the loss between ``input`` and ``target``."""
        input, target = self._prepare_input(input, target)
        sum_p, sum_target, counts = _soft_binned_calibration(input, target, self.num_bins, self.right)
        valid_bins = (counts > 0) & (counts >= self.empty_weight)
        return self._finish(sum_p, sum_target, counts, valid_bins, input, target)

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

import torch

from monai.metrics.utils import compute_voronoi_regions_fast, do_metric_reduction
from monai.utils import MetricReduction, deprecated_arg
from monai.utils.module import optional_import

from .metric import CumulativeIterationMetric

scipy_ndimage, has_scipy_ndimage = optional_import("scipy.ndimage")
cupy, has_cupy = optional_import("cupy")
cupy_ndimage, has_cupy_ndimage = optional_import("cupyx.scipy.ndimage")


__all__ = ["DiceMetric", "compute_dice", "DiceHelper"]


class DiceMetric(CumulativeIterationMetric):
    """
    Computes Dice score for a set of pairs of prediction-groundtruth labels. It supports single-channel label maps
    or multi-channel images with class segmentations per channel. This allows the computation for both multi-class
    and multi-label tasks.

    If either prediction ``y_pred`` or ground truth ``y`` have shape BCHW[D], it is expected that these represent one-
    hot segmentations for C number of classes. If either shape is B1HW[D], it is expected that these are label maps
    and the number of classes must be specified by the ``num_classes`` parameter. In either case for either inputs,
    this metric applies no activations and so non-binary values will produce unexpected results if this metric is used
    for binary overlap measurement (ie. either was expected to be one-hot formatted). Soft labels are thus permitted by
    this metric. Typically this implies that raw predictions from a network must first be activated and possibly made
    into label maps, eg. for a multi-class prediction tensor softmax and then argmax should be applied over the channel
    dimensions to produce a label map.

    The ``include_background`` parameter can be set to `False` to exclude the first category (channel index 0) which
    is by convention assumed to be background. If the non-background segmentations are small compared to the total
    image size they can get overwhelmed by the signal from the background. This assumes the shape of both prediction
    and ground truth is BCHW[D].

    The `per_component=True` approach computes the Dice metric on a per-connected component basis in the ground truth segmentation,
    ensuring equal weighting for each component regardless of its size. This method eliminates biases in traditional metrics,
    providing a more balanced evaluation, particularly in scenarios where object size does not correlate with clinical relevance.
    This provides a more granular evaluation of segmentation quality, especially useful when dealing with fragmented or
    disconnected objects in the foreground.
    Note:
    - The input prediction (`y_pred`) and ground truth (`y`) must both have 2 channels (foreground/background),
    with binary segmentation (0 for background, 1 for foreground). That is, this assumes the shape of both prediction
    and ground truth is B2HW[D].
    - This method cannot be used with multiclass segmentation.
    For more information, refer to the original paper: https://arxiv.org/abs/2410.18684

    The typical execution steps of this metric class follows :py:class:`monai.metrics.metric.Cumulative`.

    Further information can be found in the official
    `MONAI Dice Overview <https://github.com/Project-MONAI/tutorials/blob/main/modules/dice_loss_metric_notes.ipynb>`.

    Example:

    .. code-block:: python

        import torch
        from monai.metrics import DiceMetric
        from monai.losses import DiceLoss
        from monai.networks import one_hot

        batch_size, n_classes, h, w = 7, 5, 128, 128

        y_pred = torch.rand(batch_size, n_classes, h, w)  # network predictions
        y_pred = torch.argmax(y_pred, 1, True)  # convert to label map

        # ground truth as label map
        y = torch.randint(0, n_classes, size=(batch_size, 1, h, w))

        dm = DiceMetric(
            reduction="mean_batch", return_with_label=True, num_classes=n_classes
        )

        raw_scores = dm(y_pred, y)
        print(dm.aggregate())

        # now compute the Dice loss which should be the same as 1 - raw_scores
        dl = DiceLoss(to_onehot_y=True, reduction="none")
        loss = dl(one_hot(y_pred, n_classes), y).squeeze()

        print(1.0 - loss)  # same as raw_scores


    Args:
        include_background: whether to include Dice computation on the first channel/category of the prediction and
            ground truth. Defaults to ``True``, use ``False`` to exclude the background class.
        reduction: defines mode of reduction to the metrics, this will only apply reduction on `not-nan` values. The
            available reduction modes are enumerated by :py:class:`monai.utils.enums.MetricReduction`. If "none", is
            selected, the metric will not do reduction.
        get_not_nans: whether to return the `not_nans` count. If True, aggregate() returns `(metric, not_nans)` where
            `not_nans` counts the number of valid values in the result, and will have the same shape.
        ignore_empty: whether to ignore empty ground truth cases during calculation. If `True`, the `NaN` value will be
            set for an empty ground truth cases, otherwise 1 will be set if the predictions of empty ground truth cases
            are also empty.
        num_classes: number of input channels (always including the background). When this is ``None``,
            ``y_pred.shape[1]`` will be used. This option is useful when both ``y_pred`` and ``y`` are
            single-channel class indices and the number of classes is not automatically inferred from data.
        return_with_label: whether to return the metrics with label, only works when reduction is "mean_batch".
            If `True`, use "label_{index}" as the key corresponding to C channels; if ``include_background`` is True,
            the index begins at "0", otherwise at "1". It can also take a list of label names.
            The outcome will then be returned as a dictionary.
        per_component: whether to compute the Dice metric per connected component. If `True`, the metric will be
            computed for each connected component in the ground truth, and then averaged. This requires binary
            segmentations with 2 channels (background + foreground) as input. This is a more fine-grained computation.

    """

    def __init__(
        self,
        include_background: bool = True,
        reduction: MetricReduction | str = MetricReduction.MEAN,
        get_not_nans: bool = False,
        ignore_empty: bool = True,
        num_classes: int | None = None,
        return_with_label: bool | list[str] = False,
        per_component: bool = False,
    ) -> None:
        super().__init__()
        self.include_background = include_background
        self.reduction = reduction
        self.get_not_nans = get_not_nans
        self.ignore_empty = ignore_empty
        self.num_classes = num_classes
        self.return_with_label = return_with_label
        self.per_component = per_component
        self.dice_helper = DiceHelper(
            include_background=self.include_background,
            reduction=MetricReduction.NONE,
            get_not_nans=False,
            apply_argmax=False,
            ignore_empty=self.ignore_empty,
            num_classes=self.num_classes,
            per_component=self.per_component,
        )

    def _compute_tensor(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Compute the dice value using ``DiceHelper``.

        Args:
            y_pred: prediction value, see class docstring for format definition.
            y: ground truth label.

        Raises:
            ValueError: when `y_pred` has fewer than three dimensions.
        """
        dims = y_pred.ndimension()
        if dims < 3:
            raise ValueError(f"y_pred should have at least 3 dimensions (batch, channel, spatial), got {dims}.")
        # compute dice (BxC) for each channel for each batch
        return self.dice_helper(y_pred=y_pred, y=y)  # type: ignore

    def aggregate(
        self, reduction: MetricReduction | str | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Execute reduction and aggregation logic for the output of `compute_dice`.

        Args:
            reduction: defines mode of reduction as enumerated in :py:class:`monai.utils.enums.MetricReduction`.
                By default this will do no reduction.
        """
        data = self.get_buffer()
        if not isinstance(data, torch.Tensor):
            raise ValueError(f"the data to aggregate must be PyTorch Tensor, got {type(data)}.")

        # do metric reduction
        f, not_nans = do_metric_reduction(data, reduction or self.reduction)
        if self.reduction == MetricReduction.MEAN_BATCH and self.return_with_label:
            _f = {}
            if isinstance(self.return_with_label, bool):
                for i, v in enumerate(f):
                    _label_key = f"label_{i + 1}" if not self.include_background else f"label_{i}"
                    _f[_label_key] = round(v.item(), 4)
            else:
                for key, v in zip(self.return_with_label, f):
                    _f[key] = round(v.item(), 4)
            f = _f
        return (f, not_nans) if self.get_not_nans else f


def compute_dice(
    y_pred: torch.Tensor,
    y: torch.Tensor,
    include_background: bool = True,
    ignore_empty: bool = True,
    num_classes: int | None = None,
    per_component: bool = False,
) -> torch.Tensor:
    """
    Computes Dice score metric for a batch of predictions. This performs the same computation as
    :py:class:`monai.metrics.DiceMetric`, which is preferrable to use over this function. For input formats, see the
    documentation for that class .

    Args:
        y_pred: input data to compute, typical segmentation model output.
        y: ground truth to compute mean dice metric.
        include_background: whether to include Dice computation on the first channel/category of the prediction and
            ground truth. Defaults to ``True``, use ``False`` to exclude the background class.
        ignore_empty: whether to ignore empty ground truth cases during calculation. If `True`, the `NaN` value will be
            set for an empty ground truth cases, otherwise 1 will be set if the predictions of empty ground truth cases
            are also empty.
        num_classes: number of input channels (always including the background). When this is ``None``,
            ``y_pred.shape[1]`` will be used. This option is useful when both ``y_pred`` and ``y`` are
            single-channel class indices and the number of classes is not automatically inferred from data.
        per_component: whether to compute the Dice metric per connected component. If `True`, the metric will be
            computed for each connected component in the ground truth, and then averaged. This requires binary
            segmentations with 2 channels (background + foreground) as input. This is a more fine-grained computation.

    Returns:
        Dice scores per batch and per class, (shape: [batch_size, num_classes]).

    """
    return DiceHelper(  # type: ignore
        include_background=include_background,
        reduction=MetricReduction.NONE,
        get_not_nans=False,
        apply_argmax=False,
        ignore_empty=ignore_empty,
        num_classes=num_classes,
        per_component=per_component,
    )(y_pred=y_pred, y=y)


class DiceHelper:
    """
    Compute Dice score between two tensors ``y_pred`` and ``y``. This is used by :py:class:`monai.metrics.DiceMetric`,
    see the documentation for that class for input formats.

    Example:

    .. code-block:: python

        import torch
        from monai.metrics import DiceHelper

        n_classes, batch_size = 5, 16
        spatial_shape = (128, 128, 128)

        y_pred = torch.rand(batch_size, n_classes, *spatial_shape).float()  # predictions
        y = torch.randint(0, n_classes, size=(batch_size, 1, *spatial_shape)).long()  # ground truth

        score, not_nans = DiceHelper(include_background=False, sigmoid=True, softmax=True)(y_pred, y)
        print(score, not_nans)

    Args:
        include_background: whether to include Dice computation on the first channel/category of the prediction and
            ground truth. Defaults to ``True``, use ``False`` to exclude the background class.
        threshold: if ``True`, ``y_pred`` will be thresholded at a value of 0.5. Defaults to False.
        apply_argmax: whether ``y_pred`` are softmax activated outputs. If True, `argmax` will be performed to
            get the discrete prediction. Defaults to the value of ``not threshold``.
        activate: if this and ``threshold` are ``True``, sigmoid activation is applied to ``y_pred`` before
            thresholding. Defaults to False.
        get_not_nans: whether to return the number of not-nan values.
        reduction: defines mode of reduction to the metrics, this will only apply reduction on `not-nan` values. The
            available reduction modes are enumerated by :py:class:`monai.utils.enums.MetricReduction`. If "none", is
            selected, the metric will not do reduction.
        ignore_empty: whether to ignore empty ground truth cases during calculation. If `True`, the `NaN` value will be
            set for an empty ground truth cases, otherwise 1 will be set if the predictions of empty ground truth cases
            are also empty.
        num_classes: number of input channels (always including the background). When this is ``None``,
            ``y_pred.shape[1]`` will be used. This option is useful when both ``y_pred`` and ``y`` are
            single-channel class indices and the number of classes is not automatically inferred from data.
        per_component: whether to compute the Dice metric per connected component. If `True`, the metric will be
            computed for each connected component in the ground truth, and then averaged. This requires binary
            segmentations with 2 channels (background + foreground) as input. This is a more fine-grained computation.
    """

    @deprecated_arg("softmax", "1.5", "1.7", "Use `apply_argmax` instead.", new_name="apply_argmax")
    @deprecated_arg("sigmoid", "1.5", "1.7", "Use `threshold` instead.", new_name="threshold")
    def __init__(
        self,
        include_background: bool | None = None,
        threshold: bool = False,
        apply_argmax: bool | None = None,
        activate: bool = False,
        get_not_nans: bool = True,
        reduction: MetricReduction | str = MetricReduction.MEAN_BATCH,
        ignore_empty: bool = True,
        num_classes: int | None = None,
        sigmoid: bool | None = None,
        softmax: bool | None = None,
        per_component: bool = False,
    ) -> None:
        # handling deprecated arguments
        if sigmoid is not None:
            threshold = sigmoid
        if softmax is not None:
            apply_argmax = softmax

        self.threshold = threshold
        self.reduction = reduction
        self.get_not_nans = get_not_nans
        self.include_background = threshold if include_background is None else include_background
        self.apply_argmax = not threshold if apply_argmax is None else apply_argmax
        self.activate = activate
        self.ignore_empty = ignore_empty
        self.num_classes = num_classes
        self.per_component = per_component

    def compute_cc_dice(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute per-component Dice for a single batch item.

        Args:
            y_pred (torch.Tensor): Predictions with shape (1, 2, D, H, W) or (1, 2, H, W).
            y (torch.Tensor): Ground truth with shape (1, 2, D, H, W) or (1, 2, H, W).

        Returns:
            torch.Tensor: Mean Dice over connected components.
        """
        if y_pred.ndim == y.ndim:
            y_pred_idx = torch.argmax(y_pred, dim=1)
            y_idx = torch.argmax(y, dim=1)
        else:
            y_pred_idx = y_pred
            y_idx = y
        if y_idx[0].sum() == 0:
            if self.ignore_empty:
                data = torch.tensor(float("nan"), device=y_idx.device)
            elif y_pred_idx.sum() == 0:
                data = torch.tensor(1.0, device=y_idx.device)
            else:
                data = torch.tensor(0.0, device=y_idx.device)
        else:
            cc_assignment = compute_voronoi_regions_fast(y_idx[0])
            if cc_assignment.device != y_idx.device:
                cc_assignment = cc_assignment.to(y_idx.device)
            uniq, inv = torch.unique(cc_assignment.view(-1), return_inverse=True)
            nof_components = uniq.numel()
            code = (y_idx.view(-1) << 1) | y_pred_idx.view(-1)
            idx = (inv << 2) | code
            hist = torch.bincount(idx, minlength=nof_components * 4).reshape(-1, 4)
            _, fp, fn, tp = hist[:, 0], hist[:, 1], hist[:, 2], hist[:, 3]
            denom = 2 * tp + fp + fn
            dice_scores = torch.where(
                denom > 0, (2 * tp).float() / denom.float(), torch.tensor(1.0, device=denom.device)
            )
            data = dice_scores.unsqueeze(-1)
            data = torch.nan_to_num(data)
        data = data.reshape(-1, 1)
        return torch.stack([data.mean()])

    def compute_channel(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the dice metric for binary inputs which have only spatial dimensions. This method is called separately
        for each batch item and for each channel of those items.

        Args:
            y_pred: input predictions with shape HW[D].
            y: ground truth with shape HW[D].
        """
        y_o = torch.sum(y)
        if y_o > 0:
            return (2.0 * torch.sum(torch.masked_select(y, y_pred))) / (y_o + torch.sum(y_pred))
        if self.ignore_empty:
            return torch.tensor(float("nan"), device=y_o.device)
        denorm = y_o + torch.sum(y_pred)
        if denorm <= 0:
            return torch.tensor(1.0, device=y_o.device)
        return torch.tensor(0.0, device=y_o.device)

    def __call__(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the metric for the given prediction and ground truth.

        Args:
            y_pred: input predictions with shape (batch_size, num_classes or 1, spatial_dims...).
                the number of channels is inferred from ``y_pred.shape[1]`` when ``num_classes is None``.
            y: ground truth with shape (batch_size, num_classes or 1, spatial_dims...).

        Raises:
            ValueError: when the shapes of `y_pred` and `y` are not compatible for the per-component computation.
        """
        _apply_argmax, _threshold = self.apply_argmax, self.threshold
        if self.num_classes is None:
            n_pred_ch = y_pred.shape[1]  # y_pred is in one-hot format or multi-channel scores
        else:
            n_pred_ch = self.num_classes
            if y_pred.shape[1] == 1 and self.num_classes > 1:  # y_pred is single-channel class indices
                _apply_argmax = _threshold = False

        if _apply_argmax and n_pred_ch > 1:
            y_pred = torch.argmax(y_pred, dim=1, keepdim=True)

        elif _threshold:
            if self.activate:
                y_pred = torch.sigmoid(y_pred)
            y_pred = y_pred > 0.5

        if self.per_component:
            if y_pred.ndim not in (4, 5) or y.ndim not in (4, 5) or y_pred.shape[1] != 2 or y.shape[1] != 2:
                same_rank = y_pred.ndim == y.ndim and y_pred.ndim in (4, 5)
                binary_channels = y_pred.shape[1] == 2 and y.shape[1] == 2
                same_shape = y_pred.shape == y.shape
                if not (same_rank and binary_channels and same_shape):
                    raise ValueError(
                        "per_component requires matching 4D/5D binary tensors "
                        "(B, 2, H, W) or (B, 2, D, H, W). "
                        f"Got y_pred={tuple(y_pred.shape)}, y={tuple(y.shape)}."
                    )

        first_ch = 0 if self.include_background and not self.per_component else 1
        data = []
        for b in range(y_pred.shape[0]):
            if self.per_component:
                data.append(self.compute_cc_dice(y_pred=y_pred[b].unsqueeze(0), y=y[b].unsqueeze(0)).reshape(-1))
                continue
            c_list = []
            for c in range(first_ch, n_pred_ch) if n_pred_ch > 1 else [1]:
                x_pred = (y_pred[b, 0] == c) if (y_pred.shape[1] == 1) else y_pred[b, c].bool()
                x = (y[b, 0] == c) if (y.shape[1] == 1) else y[b, c]
                c_list.append(self.compute_channel(x_pred, x))
            data.append(torch.stack(c_list))

        data = torch.stack(data, dim=0).contiguous()  # type: ignore

        f, not_nans = do_metric_reduction(data, self.reduction)  # type: ignore
        return (f, not_nans) if self.get_not_nans else f

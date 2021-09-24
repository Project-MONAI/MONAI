# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
from typing import Callable, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

from monai.losses.focal_loss import FocalLoss
from monai.losses.spatial_mask import MaskedLoss
from monai.networks import one_hot
from monai.utils import LossReduction, Weight, look_up_option


class DiceLoss(_Loss):
    """
    Compute average Dice loss between two tensors. It can support both multi-classes and multi-labels tasks.
    Input logits `input` (BNHW[D] where N is number of classes) is compared with ground truth `target` (BNHW[D]).
    Axis N of `input` is expected to have logit predictions for each class rather than being image channels,
    while the same axis of `target` can be 1 or N (one-hot format). The `smooth_nr` and `smooth_dr` parameters are
    values added to the intersection and union components of the inter-over-union calculation to smooth results
    respectively, these values should be small. The `include_background` class attribute can be set to False for
    an instance of DiceLoss to exclude the first category (channel index 0) which is by convention assumed to be
    background. If the non-background segmentations are small compared to the total image size they can get
    overwhelmed by the signal from the background so excluding it in such cases helps convergence.

    Milletari, F. et. al. (2016) V-Net: Fully Convolutional Neural Networks forVolumetric Medical Image Segmentation, 3DV, 2016.

    """

    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = False,
        softmax: bool = False,
        other_act: Optional[Callable] = None,
        squared_pred: bool = False,
        jaccard: bool = False,
        reduction: Union[LossReduction, str] = LossReduction.MEAN,
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
        batch: bool = False,
        pixelwise: bool = False,
    ) -> None:
        """
        Args:
            include_background: if False, channel index 0 (background category) is excluded from the calculation.
            to_onehot_y: whether to convert `y` into the one-hot format. Defaults to False.
            sigmoid: if True, apply a sigmoid function to the prediction.
            softmax: if True, apply a softmax function to the prediction.
            other_act: if don't want to use `sigmoid` or `softmax`, use other callable function to execute
                other activation layers, Defaults to ``None``. for example:
                `other_act = torch.tanh`.
            squared_pred: use squared versions of targets and predictions in the denominator or not.
            jaccard: compute Jaccard Index (soft IoU) instead of dice or not.
            reduction: {``"none"``, ``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.

                - ``"none"``: no reduction will be applied.
                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.

            smooth_nr: a small constant added to the numerator to avoid zero.
            smooth_dr: a small constant added to the denominator to avoid nan.
            batch: whether to sum the intersection and union areas over the batch dimension before the dividing.
                Defaults to False, a Dice loss value is computed independently from each item in the batch
                before any `reduction`.

        Raises:
            TypeError: When ``other_act`` is not an ``Optional[Callable]``.
            ValueError: When more than 1 of [``sigmoid=True``, ``softmax=True``, ``other_act is not None``].
                Incompatible values.

        """
        super().__init__(reduction=LossReduction(reduction).value)
        if other_act is not None and not callable(other_act):
            raise TypeError(f"other_act must be None or callable but is {type(other_act).__name__}.")
        if int(sigmoid) + int(softmax) + int(other_act is not None) > 1:
            raise ValueError("Incompatible values: more than 1 of [sigmoid=True, softmax=True, other_act is not None].")
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.other_act = other_act
        self.squared_pred = squared_pred
        self.jaccard = jaccard
        self.smooth_nr = float(smooth_nr)
        self.smooth_dr = float(smooth_dr)
        self.batch = batch
        # pixelwise: bool = False,
        self.pixelwise = pixelwise
        if pixelwise:
            if self.reduction != LossReduction.NONE.value:
                raise ValueError('Can only compute pixelwise loss when reduction is "none"')

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD], where N is the number of classes.
            target: the shape should be BNH[WD] or B1H[WD], where N is the number of classes.

        Raises:
            AssertionError: When input and target (after one hot transform if set)
                have different shapes.
            ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].

        Example:
            >>> import torch
            >>> from monai.losses.dice import DiceLoss
            >>> B, C, H, W = 7, 5, 3, 2
            >>> input = torch.rand(B, C, H, W)
            >>> target_idx = torch.randint(low=0, high=C - 1, size=(B, H, W)).long()
            >>> target = one_hot(target_idx[:, None, ...], num_classes=C)
            >>> self = DiceLoss(reduction='none', pixelwise=True)
            >>> loss = self(input, target)
            >>> assert loss.shape == input.shape

            >>> # Original reduction=None behavior the spacetime dimensions
            >>> # are always reduced
            >>> self = DiceLoss(reduction='none', pixelwise=False, batch=False)
            >>> loss = self(input, target)
            >>> assert tuple(loss.shape) == (B, C, 1, 1)
            >>> self = DiceLoss(reduction='none', pixelwise=False, batch=True)
            >>> loss = self(input, target)
            >>> assert tuple(loss.shape) == (1, C, 1, 1)

            >>> # Test that pixelwise variants of reduce=none correspond with a reduction mode
            >>> r0 = DiceLoss(reduction='sum', batch=False)(input, target)
            >>> r1 = DiceLoss(reduction='none', batch=False, pixelwise=True)(input, target).sum()
            >>> r2 = DiceLoss(reduction='none', batch=False, pixelwise=False)(input, target).sum()
            >>> assert torch.allclose(r0, r1, rtol=1e-3, atol=1e-6)
            >>> assert torch.allclose(r0, r2, rtol=1e-3, atol=1e-6)

            >>> r0 = DiceLoss(reduction='sum', batch=True)(input, target)
            >>> r1 = DiceLoss(reduction='none', batch=True, pixelwise=True)(input, target).sum()
            >>> r2 = DiceLoss(reduction='none', batch=True, pixelwise=False)(input, target).sum()
            >>> assert torch.allclose(r0, r1, rtol=1e-3, atol=1e-6)
            >>> assert torch.allclose(r0, r2, rtol=1e-3, atol=1e-6)

            >>> r0 = DiceLoss(reduction='mean', batch=False)(input, target)
            >>> r1 = DiceLoss(reduction='none', batch=False, pixelwise=True)(input, target).sum((2, 3)).mean()
            >>> r2 = DiceLoss(reduction='none', batch=False, pixelwise=False)(input, target).mean()
            >>> assert torch.allclose(r0, r1, rtol=1e-3, atol=1e-6)
            >>> assert torch.allclose(r0, r2, rtol=1e-3, atol=1e-6)

            >>> r0 = DiceLoss(reduction='mean', batch=True)(input, target)
            >>> r1 = DiceLoss(reduction='none', batch=True, pixelwise=True)(input, target).sum((0, 2, 3)).mean()
            >>> r2 = DiceLoss(reduction='none', batch=True, pixelwise=False)(input, target).mean()
            >>> assert torch.allclose(r0, r1, rtol=1e-3, atol=1e-6)
            >>> assert torch.allclose(r0, r2, rtol=1e-3, atol=1e-6)
        """
        if self.sigmoid:
            input = torch.sigmoid(input)

        n_pred_ch = input.shape[1]
        if self.softmax:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `softmax=True` ignored.")
            else:
                input = torch.softmax(input, 1)

        if self.other_act is not None:
            input = self.other_act(input)

        if self.to_onehot_y:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `to_onehot_y=True` ignored.")
            else:
                target = one_hot(target, num_classes=n_pred_ch)

        if not self.include_background:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `include_background=False` ignored.")
            else:
                # if skipping background, removing first channel
                target = target[:, 1:]
                input = input[:, 1:]

        if target.shape != input.shape:
            raise AssertionError(f"ground truth has different shape ({target.shape}) from input ({input.shape})")

        # reducing only spatial dimensions (not batch nor channels)
        reduce_axis: List[int] = torch.arange(2, len(input.shape)).tolist()
        if self.batch:
            # reducing spatial dimensions and batch
            reduce_axis = [0] + reduce_axis

        if self.squared_pred:
            target = torch.pow(target, 2)
            input = torch.pow(input, 2)

        ground_o = torch.sum(target, dim=reduce_axis, keepdim=True)
        pred_o = torch.sum(input, dim=reduce_axis, keepdim=True)

        union = ground_o + pred_o

        if self.pixelwise and self.reduction == LossReduction.NONE.value:
            intersection = target * input

            if self.jaccard:
                denominator = 2.0 * (union - intersection.sum(dim=reduce_axis, keepdim=True))
            else:
                denominator = union

            if self.batch:
                nitems = np.prod(intersection.shape[2:]) * intersection.shape[0]
            else:
                nitems = np.prod(intersection.shape[2:])

            split_smooth_nr = self.smooth_nr / nitems
            numer_split = (2.0 * intersection + split_smooth_nr)
            denom_split = (denominator + self.smooth_dr)

            lead_split = 1 / nitems
            f: torch.Tensor = lead_split - numer_split  / denom_split
        else:
            intersection = torch.sum(target * input, dim=reduce_axis, keepdim=True)

            if self.jaccard:
                denominator = 2.0 * (union - intersection)
            else:
                denominator = union

            numer = (2.0 * intersection + self.smooth_nr)
            denom = (denominator + self.smooth_dr)
            f: torch.Tensor = 1.0 - numer / denom

            if self.reduction == LossReduction.MEAN.value:
                f = torch.mean(f)  # the batch and channel average
            elif self.reduction == LossReduction.SUM.value:
                f = torch.sum(f)  # sum over the batch and channel dims
            elif self.reduction == LossReduction.NONE.value:
                pass
                # f = torch.sum(f, dim=reduce_axis)
            else:
                raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')

        return f


class MaskedDiceLoss(DiceLoss):
    """
    Add an additional `masking` process before `DiceLoss`, accept a binary mask ([0, 1]) indicating a region,
    `input` and `target` will be masked by the region: region with mask `1` will keep the original value,
    region with `0` mask will be converted to `0`. Then feed `input` and `target` to normal `DiceLoss` computation.
    This has the effect of ensuring only the masked region contributes to the loss computation and
    hence gradient calculation.

    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Args follow :py:class:`monai.losses.DiceLoss`.
        """
        super().__init__(*args, **kwargs)
        self.spatial_weighted = MaskedLoss(loss=super().forward)

    def forward(self, input: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        Args:
            input: the shape should be BNH[WD].
            target: the shape should be BNH[WD].
            mask: the shape should B1H[WD] or 11H[WD].
        """
        return self.spatial_weighted(input=input, target=target, mask=mask)


class GeneralizedDiceLoss(_Loss):
    """
    Compute the generalised Dice loss defined in:

        Sudre, C. et. al. (2017) Generalised Dice overlap as a deep learning
        loss function for highly unbalanced segmentations. DLMIA 2017.

    Adapted from:
        https://github.com/NifTK/NiftyNet/blob/v0.6.0/niftynet/layer/loss_segmentation.py#L279
    """

    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = False,
        softmax: bool = False,
        other_act: Optional[Callable] = None,
        w_type: Union[Weight, str] = Weight.SQUARE,
        reduction: Union[LossReduction, str] = LossReduction.MEAN,
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
        batch: bool = False,
        pixelwise: bool = False,
    ) -> None:
        """
        Args:
            include_background: If False channel index 0 (background category) is excluded from the calculation.
            to_onehot_y: whether to convert `y` into the one-hot format. Defaults to False.
            sigmoid: If True, apply a sigmoid function to the prediction.
            softmax: If True, apply a softmax function to the prediction.
            other_act: if don't want to use `sigmoid` or `softmax`, use other callable function to execute
                other activation layers, Defaults to ``None``. for example:
                `other_act = torch.tanh`.
            squared_pred: use squared versions of targets and predictions in the denominator or not.
            w_type: {``"square"``, ``"simple"``, ``"uniform"``}
                Type of function to transform ground truth volume to a weight factor. Defaults to ``"square"``.
            reduction: {``"none"``, ``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.

                - ``"none"``: no reduction will be applied.
                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.
            smooth_nr: a small constant added to the numerator to avoid zero.
            smooth_dr: a small constant added to the denominator to avoid nan.
            batch: whether to sum the intersection and union areas over the batch dimension before the dividing.
                Defaults to False, intersection over union is computed from each item in the batch.

        Raises:
            TypeError: When ``other_act`` is not an ``Optional[Callable]``.
            ValueError: When more than 1 of [``sigmoid=True``, ``softmax=True``, ``other_act is not None``].
                Incompatible values.

        """
        super().__init__(reduction=LossReduction(reduction).value)
        if other_act is not None and not callable(other_act):
            raise TypeError(f"other_act must be None or callable but is {type(other_act).__name__}.")
        if int(sigmoid) + int(softmax) + int(other_act is not None) > 1:
            raise ValueError("Incompatible values: more than 1 of [sigmoid=True, softmax=True, other_act is not None].")

        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.other_act = other_act

        self.w_type = look_up_option(w_type, Weight)

        self.smooth_nr = float(smooth_nr)
        self.smooth_dr = float(smooth_dr)
        self.batch = batch

        self.pixelwise = pixelwise
        if pixelwise:
            if self.reduction != LossReduction.NONE.value:
                raise ValueError('Can only compute pixelwise loss when reduction is "none"')

    def w_func(self, grnd):
        if self.w_type == Weight.SIMPLE:
            return torch.reciprocal(grnd)
        if self.w_type == Weight.SQUARE:
            return torch.reciprocal(grnd * grnd)
        return torch.ones_like(grnd)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD].
            target: the shape should be BNH[WD].

        Raises:
            ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].

        Example:
            >>> import torch
            >>> from monai.losses.dice import GeneralizedDiceLoss
            >>> B, C, H, W = 7, 5, 3, 2
            >>> input = torch.rand(B, C, H, W)
            >>> target_idx = torch.randint(low=0, high=C - 1, size=(B, H, W)).long()
            >>> target = one_hot(target_idx[:, None, ...], num_classes=C)
            >>> self = GeneralizedDiceLoss(reduction='none', pixelwise=True)
            >>> loss = self(input, target)
            >>> assert loss.shape == input.shape

            >>> # Original reduction=None behavior the spacetime dimensions
            >>> # are always reduced
            >>> self = GeneralizedDiceLoss(reduction='none', pixelwise=False, batch=False)
            >>> loss = self(input, target)
            >>> assert tuple(loss.shape) == (B, 1, 1, 1)
            >>> self = GeneralizedDiceLoss(reduction='none', pixelwise=False, batch=True)
            >>> loss = self(input, target)
            >>> assert tuple(loss.shape) == (1, C, 1, 1)

            >>> # Test that pixelwise variants of reduce=none correspond with a reduction mode
            >>> r0 = GeneralizedDiceLoss(reduction='sum', batch=False)(input, target)
            >>> r1 = GeneralizedDiceLoss(reduction='none', batch=False, pixelwise=True)(input, target).sum()
            >>> r2 = GeneralizedDiceLoss(reduction='none', batch=False, pixelwise=False)(input, target).sum()
            >>> assert torch.allclose(r0, r1, rtol=1e-3, atol=1e-6)
            >>> assert torch.allclose(r0, r2, rtol=1e-3, atol=1e-6)

            >>> r0 = GeneralizedDiceLoss(reduction='sum', batch=True)(input, target)
            >>> r1 = GeneralizedDiceLoss(reduction='none', batch=True, pixelwise=True)(input, target).sum()
            >>> r2 = GeneralizedDiceLoss(reduction='none', batch=True, pixelwise=False)(input, target).sum()
            >>> assert torch.allclose(r0, r1, rtol=1e-3, atol=1e-6)
            >>> assert torch.allclose(r0, r2, rtol=1e-3, atol=1e-6)

            >>> r0 = GeneralizedDiceLoss(reduction='mean', batch=False)(input, target)
            >>> r1 = GeneralizedDiceLoss(reduction='none', batch=False, pixelwise=True)(input, target).sum((1, 2, 3)).mean()
            >>> r2 = GeneralizedDiceLoss(reduction='none', batch=False, pixelwise=False)(input, target).mean()
            >>> assert torch.allclose(r0, r1, rtol=1e-3, atol=1e-6)
            >>> assert torch.allclose(r0, r2, rtol=1e-3, atol=1e-6)

            >>> r0 = GeneralizedDiceLoss(reduction='mean', batch=True)(input, target)
            >>> r1 = GeneralizedDiceLoss(reduction='none', batch=True, pixelwise=True)(input, target).sum((0, 2, 3)).mean()
            >>> r2 = GeneralizedDiceLoss(reduction='none', batch=True, pixelwise=False)(input, target).mean()
            >>> assert torch.allclose(r0, r1, rtol=1e-3, atol=1e-6)
            >>> assert torch.allclose(r0, r2, rtol=1e-3, atol=1e-6)
        """
        if self.sigmoid:
            input = torch.sigmoid(input)
        n_pred_ch = input.shape[1]
        if self.softmax:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `softmax=True` ignored.")
            else:
                input = torch.softmax(input, 1)

        if self.other_act is not None:
            input = self.other_act(input)

        if self.to_onehot_y:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `to_onehot_y=True` ignored.")
            else:
                target = one_hot(target, num_classes=n_pred_ch)

        if not self.include_background:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `include_background=False` ignored.")
            else:
                # if skipping background, removing first channel
                target = target[:, 1:]
                input = input[:, 1:]

        if target.shape != input.shape:
            raise AssertionError(f"ground truth has differing shape ({target.shape}) from input ({input.shape})")

        # reducing only spatial dimensions (not batch nor channels)
        reduce_axis: List[int] = torch.arange(2, len(input.shape)).tolist()
        if self.batch:
            reduce_axis = [0] + reduce_axis

        # The union will be part of the denominator
        ground_o = torch.sum(target, reduce_axis, keepdim=True)
        pred_o = torch.sum(input, reduce_axis, keepdim=True)
        union = ground_o + pred_o

        # Number of true voxels for each category in the truth
        true_hist = torch.sum(target, reduce_axis, keepdim=True)
        w = self.w_func(true_hist.float())
        for b in w:
            infs = torch.isinf(b)
            b[infs] = 0.0
            b[infs] = torch.max(b)

        if self.pixelwise and self.reduction == LossReduction.NONE.value:
            # The trick to reduce=none is to not reduce the numerator
            # The computations are somewhat redundant and slower as compared
            # to when reduce is mean or sum
            intersection = target * input

            # Weight the numerator and denominator
            w_intersection = (intersection * w)
            w_union = (union * w).sum(0 if self.batch else 1, keepdim=True)

            # Split the numerator smooth term across voxels
            split_smooth_nr = self.smooth_nr / w_intersection.numel()

            numer_split = (2.0 * w_intersection + split_smooth_nr)
            denom_split = w_union + self.smooth_dr

            if self.batch:
                nitems = np.prod(numer_split.shape[2:]) * numer_split.shape[0]
            else:
                nitems = np.prod(numer_split.shape[1:])

            lead_split = 1 / nitems
            f: torch.Tensor = lead_split - numer_split / denom_split
        else:
            # When reduction is not None, we can be more efficient
            intersection = torch.sum(target * input, reduce_axis, keepdim=True)

            w_intersection = (intersection * w).sum(0 if self.batch else 1, keepdim=True)
            w_union = (union * w).sum(0 if self.batch else 1, keepdim=True)

            numer = (2.0 * w_intersection + self.smooth_nr)
            denom = w_union + self.smooth_dr
            f: torch.Tensor = 1.0 - numer / denom

            if self.reduction == LossReduction.MEAN.value:
                f = torch.mean(f)  # the batch and channel average
            elif self.reduction == LossReduction.SUM.value:
                f = torch.sum(f)  # sum over the batch and channel dims
            elif self.reduction == LossReduction.NONE.value:
                pass
                # f = torch.sum(f, dim=reduce_axis)
            else:
                raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')

        return f


class GeneralizedWassersteinDiceLoss(_Loss):
    """
    Compute the generalized Wasserstein Dice Loss defined in:

        Fidon L. et al. (2017) Generalised Wasserstein Dice Score for Imbalanced Multi-class
        Segmentation using Holistic Convolutional Networks. BrainLes 2017.

    Or its variant (use the option weighting_mode="GDL") defined in the Appendix of:

        Tilborghs, S. et al. (2020) Comparative study of deep learning methods for the automatic
        segmentation of lung, lesion and lesion type in CT scans of COVID-19 patients.
        arXiv preprint arXiv:2007.15546

    Adapted from:
        https://github.com/LucasFidon/GeneralizedWassersteinDiceLoss
    """

    def __init__(
        self,
        dist_matrix: Union[np.ndarray, torch.Tensor],
        weighting_mode: str = "default",
        reduction: Union[LossReduction, str] = LossReduction.MEAN,
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
        pixelwise: bool = False,
    ) -> None:
        """
        Args:
            dist_matrix: 2d tensor or 2d numpy array; matrix of distances between the classes.
            It must have dimension C x C where C is the number of classes.
            weighting_mode: {``"default"``, ``"GDL"``}
                Specifies how to weight the class-specific sum of errors.
                Default to ``"default"``.

                - ``"default"``: (recommended) use the original weighting method as in:
                    Fidon L. et al. (2017) Generalised Wasserstein Dice Score for Imbalanced Multi-class
                    Segmentation using Holistic Convolutional Networks. BrainLes 2017.
                - ``"GDL"``: use a GDL-like weighting method as in the Appendix of:
                    Tilborghs, S. et al. (2020) Comparative study of deep learning methods for the automatic
                    segmentation of lung, lesion and lesion type in CT scans of COVID-19 patients.
                    arXiv preprint arXiv:2007.15546
            reduction: {``"none"``, ``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.

                - ``"none"``: no reduction will be applied.
                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.
            smooth_nr: a small constant added to the numerator to avoid zero.
            smooth_dr: a small constant added to the denominator to avoid nan.

        Raises:
            ValueError: When ``dist_matrix`` is not a square matrix.

        Example:
            .. code-block:: python

                import torch
                import numpy as np
                from monai.losses import GeneralizedWassersteinDiceLoss

                # Example with 3 classes (including the background: label 0).
                # The distance between the background class (label 0) and the other classes is the maximum, equal to 1.
                # The distance between class 1 and class 2 is 0.5.
                dist_mat = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 0.5], [1.0, 0.5, 0.0]], dtype=np.float32)
                wass_loss = GeneralizedWassersteinDiceLoss(dist_matrix=dist_mat)

                pred_score = torch.tensor([[1000, 0, 0], [0, 1000, 0], [0, 0, 1000]], dtype=torch.float32)
                grnd = torch.tensor([0, 1, 2], dtype=torch.int64)
                wass_loss(pred_score, grnd)  # 0

        """
        super().__init__(reduction=LossReduction(reduction).value)

        if dist_matrix.shape[0] != dist_matrix.shape[1]:
            raise ValueError(f"dist_matrix must be C x C, got {dist_matrix.shape[0]} x {dist_matrix.shape[1]}.")

        if weighting_mode not in ["default", "GDL"]:
            raise ValueError("weighting_mode must be either 'default' or 'GDL, got %s." % weighting_mode)

        self.m = dist_matrix
        if isinstance(self.m, np.ndarray):
            self.m = torch.from_numpy(self.m)
        if torch.max(self.m) != 1:
            self.m = self.m / torch.max(self.m)
        self.alpha_mode = weighting_mode
        self.num_classes = self.m.size(0)
        self.smooth_nr = float(smooth_nr)
        self.smooth_dr = float(smooth_dr)

        self.pixelwise = pixelwise
        if pixelwise:
            if self.reduction != LossReduction.NONE.value:
                raise ValueError('Can only compute pixelwise loss when reduction is "none"')

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD].
            target: the shape should be BNH[WD].

        Example:
            >>> from monai.losses.dice import GeneralizedWassersteinDiceLoss
            >>> import torch
            >>> B, C, H, W = 7, 5, 3, 2
            >>> input = torch.rand(B, C, H, W)

            >>> dist_matrix = 1 - torch.eye(C, C)  # this dist matrix reduces to soft dice score

            >>> target_idx = torch.randint(low=0, high=C - 1, size=(B, H, W)).long()
            >>> #target = one_hot(target_idx[:, None, ...], num_classes=C)
            >>> target = target_idx
            >>> self = GeneralizedWassersteinDiceLoss(dist_matrix, reduction='none', pixelwise=True)
            >>> loss = self(input, target)
            >>> assert tuple(loss.shape) == (B, 1, H, W)

            >>> # Original reduction=None behavior the spacetime dimensions
            >>> # are always reduced
            >>> self = GeneralizedWassersteinDiceLoss(dist_matrix, reduction='none', pixelwise=False)
            >>> loss = self(input, target)
            >>> assert tuple(loss.shape) == (B, 1, 1, 1)

            >>> # Test that pixelwise variants of reduce=none correspond with a reduction mode
            >>> r0 = GeneralizedWassersteinDiceLoss(dist_matrix, reduction='sum')(input, target)
            >>> r1 = GeneralizedWassersteinDiceLoss(dist_matrix, reduction='none', pixelwise=True)(input, target).sum()
            >>> r2 = GeneralizedWassersteinDiceLoss(dist_matrix, reduction='none', pixelwise=False)(input, target).sum()
            >>> assert torch.allclose(r0, r1, rtol=1e-3, atol=1e-6)
            >>> assert torch.allclose(r0, r2, rtol=1e-3, atol=1e-6)

            >>> r0 = GeneralizedWassersteinDiceLoss(dist_matrix, reduction='mean')(input, target)
            >>> r1 = GeneralizedWassersteinDiceLoss(dist_matrix, reduction='none', pixelwise=True)(input, target).sum((1, 2, 3)).mean()
            >>> r2 = GeneralizedWassersteinDiceLoss(dist_matrix, reduction='none', pixelwise=False)(input, target).mean()
            >>> assert torch.allclose(r0, r1, rtol=1e-3, atol=1e-6)
            >>> assert torch.allclose(r0, r2, rtol=1e-3, atol=1e-6)
            >>> assert torch.allclose(r0, r2, rtol=1e-3, atol=1e-6)

            >>> # Test that pixelwise variants of reduce=none correspond with a reduction mode
            >>> r0 = GeneralizedWassersteinDiceLoss(dist_matrix, reduction='sum', weighting_mode='GDL')(input, target)
            >>> r1 = GeneralizedWassersteinDiceLoss(dist_matrix, reduction='none', weighting_mode='GDL', pixelwise=True)(input, target).sum()
            >>> r2 = GeneralizedWassersteinDiceLoss(dist_matrix, reduction='none', weighting_mode='GDL', pixelwise=False)(input, target).sum()
            >>> assert torch.allclose(r0, r1, rtol=1e-3, atol=1e-6)
            >>> assert torch.allclose(r0, r2, rtol=1e-3, atol=1e-6)

            >>> r0 = GeneralizedWassersteinDiceLoss(dist_matrix, reduction='mean', weighting_mode='GDL')(input, target)
            >>> r1 = GeneralizedWassersteinDiceLoss(dist_matrix, reduction='none', weighting_mode='GDL', pixelwise=True)(input, target).sum((1, 2, 3)).mean()
            >>> r2 = GeneralizedWassersteinDiceLoss(dist_matrix, reduction='none', weighting_mode='GDL', pixelwise=False)(input, target).mean()
            >>> assert torch.allclose(r0, r1, rtol=1e-3, atol=1e-6)
            >>> assert torch.allclose(r0, r2, rtol=1e-3, atol=1e-6)
            >>> assert torch.allclose(r0, r2, rtol=1e-3, atol=1e-6)
        """
        # Input shape is Batch, Classes, followed by spacetime dimensions
        B, C, *ST_DIMS = input.shape
        ST = np.prod(ST_DIMS)

        # Aggregate spatial dimensions
        flat_input = input.reshape(B, C, ST)
        flat_target = target.reshape(B, ST).long()

        # Apply the softmax to the input scores map
        probs = F.softmax(flat_input, dim=1)  # [B, C, ST]

        # Compute the Wasserstein distance map
        wass_dist_map = self.wasserstein_distance_map(probs, flat_target)  # [B, ST]

        # Compute the values of alpha to use based on :attr:`self.alpha_mode`
        alpha = self._compute_alpha_generalized_true_positives(flat_target)  # [B, C]

        true_pos_split = self._compute_generalized_true_positive(alpha, flat_target, wass_dist_map)  # [B, ST]
        # Aggregate true pos over spatial dims
        true_pos = true_pos_split.sum(dim=1)  # [B]

        # Compute the numerator and denominator of the generalized Wasserstein Dice loss
        if self.alpha_mode == "GDL":
            # use GDL-style alpha weights (i.e. normalize by the volume of each class)
            denom_split = self._compute_denominator(alpha, flat_target, wass_dist_map)  # [B, ST]
            denom = denom_split.sum(dim=1)  # [B]
        else:  # default: as in the original paper
            all_error = torch.sum(wass_dist_map, dim=1)  # [B]
            denom = 2 * true_pos + all_error  # [B]

        if self.pixelwise and self.reduction == LossReduction.NONE.value:
            # Dont reduce over the spatial resolution
            smooth_nr_split = self.smooth_nr / ST
            numer_split = (2.0 * true_pos_split + smooth_nr_split)
            wass_dice_split: torch.Tensor = numer_split / (denom[:, None] + self.smooth_dr)
            lead_split = 1 / ST
            wass_dice_loss_flat = lead_split - wass_dice_split
            # reshape back to spatial dims, categories are always reduced.
            wass_dice_loss = wass_dice_loss_flat.view(B, 1, *ST_DIMS)
        else:
            # Compute the final loss
            numer = (2.0 * true_pos + self.smooth_nr)
            wass_dice = numer / (denom + self.smooth_dr)
            wass_dice_loss_flat = 1.0 - wass_dice

            if self.reduction == LossReduction.MEAN.value:
                wass_dice_loss: torch.Tensor = torch.mean(wass_dice_loss_flat)  # the batch and channel average
            elif self.reduction == LossReduction.SUM.value:
                wass_dice_loss: torch.Tensor = torch.sum(wass_dice_loss_flat)  # sum over the batch and channel dims
            elif self.reduction == LossReduction.NONE.value:
                # If we are not computing voxelwise loss components at least
                # make sure a none reduction maintains a broadcastable shape
                broadcast_shape = [B, 1] + ([1] * len(ST_DIMS))
                wass_dice_loss: torch.Tensor = wass_dice_loss_flat.view(broadcast_shape)
            else:
                raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')

        return wass_dice_loss

    def wasserstein_distance_map(self, flat_proba: torch.Tensor, flat_target: torch.Tensor) -> torch.Tensor:
        """
        Compute the voxel-wise Wasserstein distance between the
        flattened prediction and the flattened labels (ground_truth) with respect
        to the distance matrix on the label space M.
        This corresponds to eq. 6 in:

            Fidon L. et al. (2017) Generalised Wasserstein Dice Score for Imbalanced Multi-class
            Segmentation using Holistic Convolutional Networks. BrainLes 2017.

        Args:
            flat_proba: the probabilities of input(predicted) tensor.
            flat_target: the target tensor.
        """
        # Turn the distance matrix to a map of identical matrix
        m = torch.clone(torch.as_tensor(self.m)).to(flat_proba.device)
        m_extended = torch.unsqueeze(m, dim=0)
        m_extended = torch.unsqueeze(m_extended, dim=3)
        m_extended = m_extended.expand((flat_proba.size(0), m_extended.size(1), m_extended.size(2), flat_proba.size(2)))

        # Expand the feature dimensions of the target
        flat_target_extended = torch.unsqueeze(flat_target, dim=1)
        flat_target_extended = flat_target_extended.expand(
            (flat_target.size(0), m_extended.size(1), flat_target.size(1))
        )
        flat_target_extended = torch.unsqueeze(flat_target_extended, dim=1)

        # Extract the vector of class distances for the ground-truth label at each voxel
        m_extended = torch.gather(m_extended, dim=1, index=flat_target_extended)
        m_extended = torch.squeeze(m_extended, dim=1)

        # Compute the wasserstein distance map
        wasserstein_map = m_extended * flat_proba

        # Sum over the classes
        wasserstein_map = torch.sum(wasserstein_map, dim=1)
        return wasserstein_map

    def _compute_generalized_true_positive(
        self, alpha: torch.Tensor, flat_target: torch.Tensor, wass_dist_map: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            alpha: generalised number of true positives of target class.
                shape is (B, C)

            flat_target: the target tensor of class indexes.
                shape is (B, ST)

            wass_dist_map: the map obtained from the above function.
                shape is (B, ST)
        """
        # Extend alpha to a map and select value at each voxel according to flat_target
        alpha_lut = torch.unsqueeze(alpha, dim=2)
        alpha_lut = alpha_lut.expand((flat_target.size(0), self.num_classes, flat_target.size(1)))  # [B, C, ST]
        flat_target_extended = torch.unsqueeze(flat_target, dim=1)
        alpha_extended = torch.gather(alpha_lut, index=flat_target_extended, dim=1)  # [B, 1, ST]
        wass_sim = (1.0 - wass_dist_map)[None, :]  # [1, B, ST]
        prod = alpha_extended * wass_sim  # [B, B, ST]
        true_pos_split = torch.sum(prod, dim=[1])  # [B, ST]
        return true_pos_split

    def _compute_denominator(
        self, alpha: torch.Tensor, flat_target: torch.Tensor, wass_dist_map: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            alpha: generalised number of true positives of target class.
            flat_target: the target tensor.
            wass_dist_map: the map obtained from the above function.
        """
        # Extend alpha to a map and select value at each voxel according to flat_target
        alpha_lut = torch.unsqueeze(alpha, dim=2)
        alpha_lut = alpha_lut.expand((flat_target.size(0), self.num_classes, flat_target.size(1)))
        flat_target_extended = torch.unsqueeze(flat_target, dim=1)
        alpha_extended = torch.gather(alpha_lut, index=flat_target_extended, dim=1)  # [B, 1, ST]
        wass_2sim = (2.0 - wass_dist_map)[None, :]  # [1, B, ST]
        prod = alpha_extended * wass_2sim  # [B, B, ST]
        denom_split  = torch.sum(prod, dim=[1])  # [B, ST]
        return denom_split

    def _compute_alpha_generalized_true_positives(self, flat_target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            flat_target: the target tensor.
        """
        alpha: torch.Tensor = torch.ones((flat_target.size(0), self.num_classes)).float().to(flat_target.device)
        if self.alpha_mode == "GDL":  # GDL style
            # use GDL-style alpha weights (i.e. normalize by the volume of each class)
            # contrary to the original definition we also use alpha in the "generalized all error".
            one_hot_f = F.one_hot(flat_target, num_classes=self.num_classes).permute(0, 2, 1).float()
            volumes = torch.sum(one_hot_f, dim=2)
            alpha = 1.0 / (volumes + 1.0)
        else:
            # default, i.e. like in the original paper
            # (i.e. alpha=1 for all foreground classes and 0 for the background).
            # Compute the generalised number of true positives
            # TODO: parametarize background index
            alpha[:, 0] = 0.0
        return alpha


class DiceCELoss(_Loss):
    """
    Compute both Dice loss and Cross Entropy Loss, and return the weighted sum of these two losses.
    The details of Dice loss is shown in ``monai.losses.DiceLoss``.
    The details of Cross Entropy Loss is shown in ``torch.nn.CrossEntropyLoss``. In this implementation,
    two deprecated parameters ``size_average`` and ``reduce``, and the parameter ``ignore_index`` are
    not supported.

    """

    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = False,
        softmax: bool = False,
        other_act: Optional[Callable] = None,
        squared_pred: bool = False,
        jaccard: bool = False,
        reduction: str = "mean",
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
        batch: bool = False,
        ce_weight: Optional[torch.Tensor] = None,
        lambda_dice: float = 1.0,
        lambda_ce: float = 1.0,
        balance_broadcast: bool = False,
    ) -> None:
        """
        Args:
            ``ce_weight`` and ``lambda_ce`` are only used for cross entropy loss.
            ``reduction`` is used for both losses and other parameters are only used for dice loss.

            include_background: if False channel index 0 (background category) is excluded from the calculation.
            to_onehot_y: whether to convert `y` into the one-hot format. Defaults to False.
            sigmoid: if True, apply a sigmoid function to the prediction, only used by the `DiceLoss`,
                don't need to specify activation function for `CrossEntropyLoss`.
            softmax: if True, apply a softmax function to the prediction, only used by the `DiceLoss`,
                don't need to specify activation function for `CrossEntropyLoss`.
            other_act: if don't want to use `sigmoid` or `softmax`, use other callable function to execute
                other activation layers, Defaults to ``None``. for example: `other_act = torch.tanh`.
                only used by the `DiceLoss`, don't need to specify activation function for `CrossEntropyLoss`.
            squared_pred: use squared versions of targets and predictions in the denominator or not.
            jaccard: compute Jaccard Index (soft IoU) instead of dice or not.
            reduction: {``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``. The dice loss should
                as least reduce the spatial dimensions, which is different from cross entropy loss, thus here
                the ``none`` option cannot be used.

                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.

            smooth_nr: a small constant added to the numerator to avoid zero.
            smooth_dr: a small constant added to the denominator to avoid nan.
            batch: whether to sum the intersection and union areas over the batch dimension before the dividing.
                Defaults to False, a Dice loss value is computed independently from each item in the batch
                before any `reduction`.
            ce_weight: a rescaling weight given to each class for cross entropy loss.
                See ``torch.nn.CrossEntropyLoss()`` for more information.
            lambda_dice: the trade-off weight value for dice loss. The value should be no less than 0.0.
                Defaults to 1.0.
            lambda_ce: the trade-off weight value for cross entropy loss. The value should be no less than 0.0.
                Defaults to 1.0.

        """
        super().__init__()
        self.dice = DiceLoss(
            include_background=include_background,
            to_onehot_y=to_onehot_y,
            sigmoid=sigmoid,
            softmax=softmax,
            other_act=other_act,
            squared_pred=squared_pred,
            jaccard=jaccard,
            reduction=reduction,
            smooth_nr=smooth_nr,
            smooth_dr=smooth_dr,
            batch=batch,
        )
        self.cross_entropy = nn.CrossEntropyLoss(
            weight=ce_weight,
            reduction=reduction,
        )
        if lambda_dice < 0.0:
            raise ValueError("lambda_dice should be no less than 0.0.")
        if lambda_ce < 0.0:
            raise ValueError("lambda_ce should be no less than 0.0.")
        self.lambda_dice = lambda_dice
        self.lambda_ce = lambda_ce

        self.reduction = reduction

        # Better Name?
        self.balance_broadcast = balance_broadcast
        if balance_broadcast:
            if self.reduction != LossReduction.NONE.value:
                raise ValueError('Can only compute balance_broadcast loss when reduction is "none"')

    def ce(self, input: torch.Tensor, target: torch.Tensor):
        """
        Compute CrossEntropy loss for the input and target.
        Will remove the channel dim according to PyTorch CrossEntropyLoss:
        https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html?#torch.nn.CrossEntropyLoss.

        """
        n_pred_ch, n_target_ch = input.shape[1], target.shape[1]
        if n_pred_ch == n_target_ch:
            # target is in the one-hot format, convert to BH[WD] format to calculate ce loss
            target = torch.argmax(target, dim=1)
        else:
            target = torch.squeeze(target, dim=1)
        target = target.long()
        return self.cross_entropy(input, target)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD].
            target: the shape should be BNH[WD] or B1H[WD].

        Raises:
            ValueError: When number of dimensions for input and target are different.
            ValueError: When number of channels for target is neither 1 nor the same as input.

        Example:
            >>> import torch
            >>> from monai.losses.dice import DiceLoss
            >>> C = 5
            >>> input = torch.rand(7, 5, 3, 2)
            >>> target_idx = torch.randint(low=0, high=C - 1, size=(7, 3, 2)).long()
            >>> target = one_hot(target_idx[:, None, ...], num_classes=C)
            >>> self = DiceCELoss(reduction='none')
            >>> loss = self(input, target)
            >>> assert loss.shape == input.shape

            >>> # Test that pixelwise variants of reduce=none correspond with a reduction mode
            >>> r0 = DiceCELoss(reduction='sum', batch=False)(input, target)
            >>> r1 = DiceCELoss(reduction='none', batch=False, balance_broadcast=True)(input, target).sum()
            >>> print('r0 = {!r}'.format(r0))
            >>> print('r1 = {!r}'.format(r1))
            >>> assert torch.allclose(r0, r1, rtol=1e-3, atol=1e-6)

            >>> r0 = DiceCELoss(reduction='sum', batch=True)(input, target)
            >>> r1 = DiceCELoss(reduction='none', batch=True, balance_broadcast=True)(input, target).sum()
            >>> print('r0 = {!r}'.format(r0))
            >>> print('r1 = {!r}'.format(r1))
            >>> assert torch.allclose(r0, r1, rtol=1e-3, atol=1e-6)

            >>> r0 = DiceCELoss(reduction='mean', batch=False)(input, target)
            >>> r1 = DiceCELoss(reduction='none', batch=False, balance_broadcast=False)(input, target).mean()
            >>> print('r0 = {!r}'.format(r0))
            >>> print('r1 = {!r}'.format(r1))
            >>> assert torch.allclose(r0, r1, rtol=1e-3, atol=1e-6)

            >>> r0 = DiceCELoss(reduction='mean', batch=True)(input, target)
            >>> r1 = DiceCELoss(reduction='none', batch=True, balance_broadcast=False)(input, target).mean()
            >>> print('r0 = {!r}'.format(r0))
            >>> print('r1 = {!r}'.format(r1))
            >>> assert torch.allclose(r0, r1, rtol=1e-3, atol=1e-6)
        """
        if len(input.shape) != len(target.shape):
            raise ValueError("the number of dimensions for input and target should be the same.")

        dice_loss = self.dice(input, target)
        ce_loss = self.ce(input, target)

        if self.reduction == LossReduction.NONE.value:
            # Expand the class dimension for reduction=none compatability
            ce_loss = ce_loss[:, None, ...]

            # If we want to apply "mean" reduction to a "none" reduced
            # item after the fact, balance_broadcast must be False,
            # and for "sum", balance_broadcast must be True.
            if self.balance_broadcast:
                # Broadcasting will introduce duplicates of items, so we have to
                # componestate for that. This does cause "mean" reduction
                # to compoenstate
                nitems_final = np.prod(torch.broadcast_shapes(
                    dice_loss.shape, ce_loss.shape))

                nitems_dice = np.prod(dice_loss.shape)
                nitems_ce = np.prod(ce_loss.shape)

                dice_bcast_factor = (nitems_final // nitems_dice)
                ce_bcase_factor   = (nitems_final // nitems_ce)

                dice_loss = dice_loss * (1.0 / dice_bcast_factor)
                ce_loss = ce_loss * (1.0 / ce_bcase_factor)

        total_loss: torch.Tensor = self.lambda_dice * dice_loss + self.lambda_ce * ce_loss

        return total_loss


class DiceFocalLoss(_Loss):
    """
    Compute both Dice loss and Focal Loss, and return the weighted sum of these two losses.
    The details of Dice loss is shown in ``monai.losses.DiceLoss``.
    The details of Focal Loss is shown in ``monai.losses.FocalLoss``.

    """

    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = False,
        softmax: bool = False,
        other_act: Optional[Callable] = None,
        squared_pred: bool = False,
        jaccard: bool = False,
        reduction: str = "mean",
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
        batch: bool = False,
        gamma: float = 2.0,
        focal_weight: Optional[Union[Sequence[float], float, int, torch.Tensor]] = None,
        lambda_dice: float = 1.0,
        lambda_focal: float = 1.0,
        balance_broadcast: bool = False,
    ) -> None:
        """
        Args:
            ``gamma``, ``focal_weight`` and ``lambda_focal`` are only used for focal loss.
            ``include_background``, ``to_onehot_y``and ``reduction`` are used for both losses
            and other parameters are only used for dice loss.
            include_background: if False channel index 0 (background category) is excluded from the calculation.
            to_onehot_y: whether to convert `y` into the one-hot format. Defaults to False.
            sigmoid: if True, apply a sigmoid function to the prediction, only used by the `DiceLoss`,
                don't need to specify activation function for `FocalLoss`.
            softmax: if True, apply a softmax function to the prediction, only used by the `DiceLoss`,
                don't need to specify activation function for `FocalLoss`.
            other_act: if don't want to use `sigmoid` or `softmax`, use other callable function to execute
                other activation layers, Defaults to ``None``. for example: `other_act = torch.tanh`.
                only used by the `DiceLoss`, don't need to specify activation function for `FocalLoss`.
            squared_pred: use squared versions of targets and predictions in the denominator or not.
            jaccard: compute Jaccard Index (soft IoU) instead of dice or not.
            reduction: {``"none"``, ``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.

                - ``"none"``: no reduction will be applied.
                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.

            smooth_nr: a small constant added to the numerator to avoid zero.
            smooth_dr: a small constant added to the denominator to avoid nan.
            batch: whether to sum the intersection and union areas over the batch dimension before the dividing.
                Defaults to False, a Dice loss value is computed independently from each item in the batch
                before any `reduction`.
            gamma: value of the exponent gamma in the definition of the Focal loss.
            focal_weight: weights to apply to the voxels of each class. If None no weights are applied.
                The input can be a single value (same weight for all classes), a sequence of values (the length
                of the sequence should be the same as the number of classes).
            lambda_dice: the trade-off weight value for dice loss. The value should be no less than 0.0.
                Defaults to 1.0.
            lambda_focal: the trade-off weight value for focal loss. The value should be no less than 0.0.
                Defaults to 1.0.

        """
        super().__init__()
        self.dice = DiceLoss(
            include_background=include_background,
            to_onehot_y=to_onehot_y,
            sigmoid=sigmoid,
            softmax=softmax,
            other_act=other_act,
            squared_pred=squared_pred,
            jaccard=jaccard,
            reduction=reduction,
            smooth_nr=smooth_nr,
            smooth_dr=smooth_dr,
            batch=batch,
        )
        self.focal = FocalLoss(
            include_background=include_background,
            to_onehot_y=to_onehot_y,
            gamma=gamma,
            weight=focal_weight,
            reduction=reduction,
        )
        if lambda_dice < 0.0:
            raise ValueError("lambda_dice should be no less than 0.0.")
        if lambda_focal < 0.0:
            raise ValueError("lambda_focal should be no less than 0.0.")
        self.lambda_dice = lambda_dice
        self.lambda_focal = lambda_focal
        self.reduction = reduction

        # Better Name?
        self.balance_broadcast = balance_broadcast
        if balance_broadcast:
            if self.reduction != LossReduction.NONE.value:
                raise ValueError('Can only compute balance_broadcast loss when reduction is "none"')

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD]. The input should be the original logits
                due to the restriction of ``monai.losses.FocalLoss``.
            target: the shape should be BNH[WD] or B1H[WD].

        Raises:
            ValueError: When number of dimensions for input and target are different.
            ValueError: When number of channels for target is neither 1 nor the same as input.

        Example:
            >>> import torch
            >>> from monai.losses.dice import DiceLoss
            >>> C = 5
            >>> input = torch.rand(7, 5, 3, 2)
            >>> target_idx = torch.randint(low=0, high=C - 1, size=(7, 3, 2)).long()
            >>> target = one_hot(target_idx[:, None, ...], num_classes=C)
            >>> self = DiceFocalLoss(reduction='none')
            >>> loss = self(input, target)
            >>> assert loss.shape == input.shape

            >>> # Test that pixelwise variants of reduce=none correspond with a reduction mode
            >>> r0 = DiceFocalLoss(reduction='sum', batch=False)(input, target)
            >>> r1 = DiceFocalLoss(reduction='none', batch=False, balance_broadcast=True)(input, target).sum()
            >>> print('r0 = {!r}'.format(r0))
            >>> print('r1 = {!r}'.format(r1))
            >>> assert torch.allclose(r0, r1, rtol=1e-3, atol=1e-6)

            >>> r0 = DiceFocalLoss(reduction='sum', batch=True)(input, target)
            >>> r1 = DiceFocalLoss(reduction='none', batch=True, balance_broadcast=True)(input, target).sum()
            >>> print('r0 = {!r}'.format(r0))
            >>> print('r1 = {!r}'.format(r1))
            >>> assert torch.allclose(r0, r1, rtol=1e-3, atol=1e-6)

            >>> r0 = DiceFocalLoss(reduction='mean', batch=False)(input, target)
            >>> r1 = DiceFocalLoss(reduction='none', batch=False)(input, target).mean()
            >>> print('r0 = {!r}'.format(r0))
            >>> print('r1 = {!r}'.format(r1))
            >>> assert torch.allclose(r0, r1, rtol=1e-3, atol=1e-6)

            >>> r0 = DiceFocalLoss(reduction='mean', batch=True)(input, target)
            >>> r1 = DiceFocalLoss(reduction='none', batch=True)(input, target).mean()
            >>> print('r0 = {!r}'.format(r0))
            >>> print('r1 = {!r}'.format(r1))
            >>> assert torch.allclose(r0, r1, rtol=1e-3, atol=1e-6)
        """
        if len(input.shape) != len(target.shape):
            raise ValueError("the number of dimensions for input and target should be the same.")

        dice_loss = self.dice(input, target)
        focal_loss = self.focal(input, target)

        if self.reduction == LossReduction.NONE.value:
            # If we want to apply "mean" reduction to a "none" reduced
            # item after the fact, balance_broadcast must be False,
            # and for "sum", balance_broadcast must be True.
            if self.balance_broadcast:
                # Broadcasting will introduce duplicates of items, so we have to
                # componestate for that. This does cause "mean" reduction
                # to compoenstate
                nitems_final = np.prod(torch.broadcast_shapes(
                    dice_loss.shape, focal_loss.shape))

                nitems_dice = np.prod(dice_loss.shape)
                nitems_focal = np.prod(focal_loss.shape)

                dice_bcast_factor = (nitems_final // nitems_dice)
                focal_bcase_factor   = (nitems_final // nitems_focal)

                dice_loss = dice_loss * (1.0 / dice_bcast_factor)
                focal_loss = focal_loss * (1.0 / focal_bcase_factor)

        total_loss: torch.Tensor = self.lambda_dice * dice_loss + self.lambda_focal * focal_loss

        return total_loss


dice = Dice = DiceLoss
dice_ce = DiceCELoss
dice_focal = DiceFocalLoss
generalized_dice = GeneralizedDiceLoss
generalized_wasserstein_dice = GeneralizedWassersteinDiceLoss

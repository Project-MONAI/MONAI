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

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD], where N is the number of classes.
            target: the shape should be BNH[WD] or B1H[WD], where N is the number of classes.

        Raises:
            AssertionError: When input and target (after one hot transform if set)
                have different shapes.
            ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].

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

        intersection = torch.sum(target * input, dim=reduce_axis)

        if self.squared_pred:
            target = torch.pow(target, 2)
            input = torch.pow(input, 2)

        ground_o = torch.sum(target, dim=reduce_axis)
        pred_o = torch.sum(input, dim=reduce_axis)

        denominator = ground_o + pred_o

        if self.jaccard:
            denominator = 2.0 * (denominator - intersection)

        f: torch.Tensor = 1.0 - (2.0 * intersection + self.smooth_nr) / (denominator + self.smooth_dr)

        if self.reduction == LossReduction.MEAN.value:
            f = torch.mean(f)  # the batch and channel average
        elif self.reduction == LossReduction.SUM.value:
            f = torch.sum(f)  # sum over the batch and channel dims
        elif self.reduction != LossReduction.NONE.value:
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
        intersection = torch.sum(target * input, reduce_axis)

        ground_o = torch.sum(target, reduce_axis)
        pred_o = torch.sum(input, reduce_axis)

        denominator = ground_o + pred_o

        w = self.w_func(ground_o.float())
        for b in w:
            infs = torch.isinf(b)
            b[infs] = 0.0
            b[infs] = torch.max(b)

        f: torch.Tensor = 1.0 - (2.0 * (intersection * w).sum(0 if self.batch else 1) + self.smooth_nr) / (
            (denominator * w).sum(0 if self.batch else 1) + self.smooth_dr
        )

        if self.reduction == LossReduction.MEAN.value:
            f = torch.mean(f)  # the batch and channel average
        elif self.reduction == LossReduction.SUM.value:
            f = torch.sum(f)  # sum over the batch and channel dims
        elif self.reduction != LossReduction.NONE.value:
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
        super(GeneralizedWassersteinDiceLoss, self).__init__(reduction=LossReduction(reduction).value)

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

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD].
            target: the shape should be BNH[WD].

        """
        # Aggregate spatial dimensions
        flat_input = input.reshape(input.size(0), input.size(1), -1)
        flat_target = target.reshape(target.size(0), -1).long()

        # Apply the softmax to the input scores map
        probs = F.softmax(flat_input, dim=1)

        # Compute the Wasserstein distance map
        wass_dist_map = self.wasserstein_distance_map(probs, flat_target)

        # Compute the values of alpha to use
        alpha = self._compute_alpha_generalized_true_positives(flat_target)

        # Compute the numerator and denominator of the generalized Wasserstein Dice loss
        if self.alpha_mode == "GDL":
            # use GDL-style alpha weights (i.e. normalize by the volume of each class)
            # contrary to the original definition we also use alpha in the "generalized all error".
            true_pos = self._compute_generalized_true_positive(alpha, flat_target, wass_dist_map)
            denom = self._compute_denominator(alpha, flat_target, wass_dist_map)
        else:  # default: as in the original paper
            # (i.e. alpha=1 for all foreground classes and 0 for the background).
            # Compute the generalised number of true positives
            true_pos = self._compute_generalized_true_positive(alpha, flat_target, wass_dist_map)
            all_error = torch.sum(wass_dist_map, dim=1)
            denom = 2 * true_pos + all_error

        # Compute the final loss
        wass_dice: torch.Tensor = (2.0 * true_pos + self.smooth_nr) / (denom + self.smooth_dr)
        wass_dice_loss: torch.Tensor = 1.0 - wass_dice

        if self.reduction == LossReduction.MEAN.value:
            wass_dice_loss = torch.mean(wass_dice_loss)  # the batch and channel average
        elif self.reduction == LossReduction.SUM.value:
            wass_dice_loss = torch.sum(wass_dice_loss)  # sum over the batch and channel dims
        elif self.reduction != LossReduction.NONE.value:
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
        self, alpha: torch.Tensor, flat_target: torch.Tensor, wasserstein_distance_map: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            alpha: generalised number of true positives of target class.
            flat_target: the target tensor.
            wasserstein_distance_map: the map obtained from the above function.
        """
        # Extend alpha to a map and select value at each voxel according to flat_target
        alpha_extended = torch.unsqueeze(alpha, dim=2)
        alpha_extended = alpha_extended.expand((flat_target.size(0), self.num_classes, flat_target.size(1)))
        flat_target_extended = torch.unsqueeze(flat_target, dim=1)
        alpha_extended = torch.gather(alpha_extended, index=flat_target_extended, dim=1)

        return torch.sum(
            alpha_extended * (1.0 - wasserstein_distance_map),
            dim=[1, 2],
        )

    def _compute_denominator(
        self, alpha: torch.Tensor, flat_target: torch.Tensor, wasserstein_distance_map: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            alpha: generalised number of true positives of target class.
            flat_target: the target tensor.
            wasserstein_distance_map: the map obtained from the above function.
        """
        # Extend alpha to a map and select value at each voxel according to flat_target
        alpha_extended = torch.unsqueeze(alpha, dim=2)
        alpha_extended = alpha_extended.expand((flat_target.size(0), self.num_classes, flat_target.size(1)))
        flat_target_extended = torch.unsqueeze(flat_target, dim=1)
        alpha_extended = torch.gather(alpha_extended, index=flat_target_extended, dim=1)

        return torch.sum(
            alpha_extended * (2.0 - wasserstein_distance_map),
            dim=[1, 2],
        )

    def _compute_alpha_generalized_true_positives(self, flat_target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            flat_target: the target tensor.
        """
        alpha: torch.Tensor = torch.ones((flat_target.size(0), self.num_classes)).float().to(flat_target.device)
        if self.alpha_mode == "GDL":  # GDL style
            # Define alpha like in the generalized dice loss
            # i.e. the inverse of the volume of each class.
            one_hot_f = F.one_hot(flat_target, num_classes=self.num_classes).permute(0, 2, 1).float()
            volumes = torch.sum(one_hot_f, dim=2)
            alpha = 1.0 / (volumes + 1.0)
        else:  # default, i.e. like in the original paper
            # alpha weights are 0 for the background and 1 the other classes
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

        """
        if len(input.shape) != len(target.shape):
            raise ValueError("the number of dimensions for input and target should be the same.")

        dice_loss = self.dice(input, target)
        ce_loss = self.ce(input, target)
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

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD]. The input should be the original logits
                due to the restriction of ``monai.losses.FocalLoss``.
            target: the shape should be BNH[WD] or B1H[WD].

        Raises:
            ValueError: When number of dimensions for input and target are different.
            ValueError: When number of channels for target is neither 1 nor the same as input.

        """
        if len(input.shape) != len(target.shape):
            raise ValueError("the number of dimensions for input and target should be the same.")

        dice_loss = self.dice(input, target)
        focal_loss = self.focal(input, target)
        total_loss: torch.Tensor = self.lambda_dice * dice_loss + self.lambda_focal * focal_loss

        return total_loss


dice = Dice = DiceLoss
dice_ce = DiceCELoss
dice_focal = DiceFocalLoss
generalized_dice = GeneralizedDiceLoss
generalized_wasserstein_dice = GeneralizedWassersteinDiceLoss

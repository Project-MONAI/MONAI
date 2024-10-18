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

# Hausdorff loss implementation based on paper:
# https://arxiv.org/pdf/1904.10030.pdf

# Repo: https://github.com/PatRyg99/HausdorffLoss

from __future__ import annotations

import warnings
from typing import Callable

import torch
from torch.nn.modules.loss import _Loss

from monai.networks import one_hot
from monai.transforms.utils import distance_transform_edt
from monai.utils import LossReduction


class HausdorffDTLoss(_Loss):
    """
    Compute channel-wise binary Hausdorff loss based on distance transform. It can support both multi-classes and
    multi-labels tasks. The data `input` (BNHW[D] where N is number of classes) is compared with ground truth `target`
    (BNHW[D]).

    Note that axis N of `input` is expected to be logits or probabilities for each class, if passing logits as input,
    must set `sigmoid=True` or `softmax=True`, or specifying `other_act`. And the same axis of `target`
    can be 1 or N (one-hot format).

    The original paper: Karimi, D. et. al. (2019) Reducing the Hausdorff Distance in Medical Image Segmentation with
    Convolutional Neural Networks, IEEE Transactions on medical imaging, 39(2), 499-513
    """

    def __init__(
        self,
        alpha: float = 2.0,
        include_background: bool = False,
        to_onehot_y: bool = False,
        sigmoid: bool = False,
        softmax: bool = False,
        other_act: Callable | None = None,
        reduction: LossReduction | str = LossReduction.MEAN,
        batch: bool = False,
    ) -> None:
        """
        Args:
            include_background: if False, channel index 0 (background category) is excluded from the calculation.
                if the non-background segmentations are small compared to the total image size they can get overwhelmed
                by the signal from the background so excluding it in such cases helps convergence.
            to_onehot_y: whether to convert the ``target`` into the one-hot format,
                using the number of classes inferred from `input` (``input.shape[1]``). Defaults to False.
            sigmoid: if True, apply a sigmoid function to the prediction.
            softmax: if True, apply a softmax function to the prediction.
            other_act: callable function to execute other activation layers, Defaults to ``None``. for example:
                ``other_act = torch.tanh``.
            reduction: {``"none"``, ``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.

                - ``"none"``: no reduction will be applied.
                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.
            batch: whether to sum the intersection and union areas over the batch dimension before the dividing.
                Defaults to False, a loss value is computed independently from each item in the batch
                before any `reduction`.

        Raises:
            TypeError: When ``other_act`` is not an ``Optional[Callable]``.
            ValueError: When more than 1 of [``sigmoid=True``, ``softmax=True``, ``other_act is not None``].
                Incompatible values.

        """
        super().__init__(reduction=LossReduction(reduction).value)
        if other_act is not None and not callable(other_act):
            raise TypeError(f"other_act must be None or callable but is {type(other_act).__name__}.")
        if int(sigmoid) + int(softmax) > 1:
            raise ValueError("Incompatible values: more than 1 of [sigmoid=True, softmax=True, other_act is not None].")

        self.alpha = alpha
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.other_act = other_act
        self.batch = batch

    @torch.no_grad()
    def distance_field(self, img: torch.Tensor) -> torch.Tensor:
        """Generate distance transform.

        Args:
            img (np.ndarray): input mask as NCHWD or NCHW.

        Returns:
            np.ndarray: Distance field.
        """
        field = torch.zeros_like(img)

        for batch_idx in range(len(img)):
            fg_mask = img[batch_idx] > 0.5

            # For cases where the mask is entirely background or entirely foreground
            # the distance transform is not well defined for all 1s,
            # which always would happen on either foreground or background, so skip
            if fg_mask.any() and not fg_mask.all():
                fg_dist: torch.Tensor = distance_transform_edt(fg_mask)  # type: ignore
                bg_mask = ~fg_mask
                bg_dist: torch.Tensor = distance_transform_edt(bg_mask)  # type: ignore

                field[batch_idx] = fg_dist + bg_dist

        return field

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNHW[D], where N is the number of classes.
            target: the shape should be BNHW[D] or B1HW[D], where N is the number of classes.

        Raises:
            ValueError: If the input is not 2D (NCHW) or 3D (NCHWD).
            AssertionError: When input and target (after one hot transform if set)
                have different shapes.
            ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].

        Example:
            >>> import torch
            >>> from monai.losses.hausdorff_loss import HausdorffDTLoss
            >>> from monai.networks.utils import one_hot
            >>> B, C, H, W = 7, 5, 3, 2
            >>> input = torch.rand(B, C, H, W)
            >>> target_idx = torch.randint(low=0, high=C - 1, size=(B, H, W)).long()
            >>> target = one_hot(target_idx[:, None, ...], num_classes=C)
            >>> self = HausdorffDTLoss(reduction='none')
            >>> loss = self(input, target)
            >>> assert np.broadcast_shapes(loss.shape, input.shape) == input.shape
        """
        if input.dim() != 4 and input.dim() != 5:
            raise ValueError("Only 2D (NCHW) and 3D (NCHWD) supported")

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
                # If skipping background, removing first channel
                target = target[:, 1:]
                input = input[:, 1:]

        if target.shape != input.shape:
            raise AssertionError(f"ground truth has different shape ({target.shape}) from input ({input.shape})")

        device = input.device
        all_f = []
        for i in range(input.shape[1]):
            ch_input = input[:, [i]]
            ch_target = target[:, [i]]
            pred_dt = self.distance_field(ch_input.detach()).float()
            target_dt = self.distance_field(ch_target.detach()).float()

            pred_error = (ch_input - ch_target) ** 2
            distance = pred_dt**self.alpha + target_dt**self.alpha

            running_f = pred_error * distance.to(device)
            reduce_axis: list[int] = torch.arange(2, len(input.shape)).tolist()
            if self.batch:
                # reducing spatial dimensions and batch
                reduce_axis = [0] + reduce_axis
            all_f.append(running_f.mean(dim=reduce_axis, keepdim=True))
        f = torch.cat(all_f, dim=1)
        if self.reduction == LossReduction.MEAN.value:
            f = torch.mean(f)  # the batch and channel average
        elif self.reduction == LossReduction.SUM.value:
            f = torch.sum(f)  # sum over the batch and channel dims
        elif self.reduction == LossReduction.NONE.value:
            # If we are not computing voxelwise loss components at least make sure a none reduction maintains a
            # broadcastable shape
            broadcast_shape = list(f.shape[0:2]) + [1] * (len(ch_input.shape) - 2)
            f = f.view(broadcast_shape)
        else:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')

        return f


class LogHausdorffDTLoss(HausdorffDTLoss):
    """
    Compute the logarithm of the Hausdorff Distance Transform Loss.

    This class computes the logarithm of the Hausdorff Distance Transform Loss, which is based on the distance transform.
    The logarithm is computed to potentially stabilize and scale the loss values, especially when the original loss
    values are very small.

    The formula for the loss is given by:
        log_loss = log(HausdorffDTLoss + 1)

    Inherits from the HausdorffDTLoss class to utilize its distance transform computation.
    """

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the logarithm of the Hausdorff Distance Transform Loss.

        Args:
            input (torch.Tensor): The shape should be BNHW[D], where N is the number of classes.
            target (torch.Tensor): The shape should be BNHW[D] or B1HW[D], where N is the number of classes.

        Returns:
            torch.Tensor: The computed Log Hausdorff Distance Transform Loss for the given input and target.

        Raises:
            Any exceptions raised by the parent class HausdorffDTLoss.
        """
        log_loss: torch.Tensor = torch.log(super().forward(input, target) + 1)
        return log_loss

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
from collections.abc import Callable

import torch
from torch.nn.modules.loss import _Loss

from monai.networks import one_hot
from monai.transforms.utils import distance_transform_edt
from monai.utils import LossReduction

__all__ = ["BoundaryLoss"]


class BoundaryLoss(_Loss):
    """
    Compute the boundary loss for highly unbalanced segmentation.

    The boundary loss is a distance-based loss that operates on the interface between segmentation
    regions rather than on the regions themselves. This makes it particularly effective for
    highly imbalanced segmentation tasks (e.g., small lesions, thin structures), where standard
    Dice or Cross-Entropy losses struggle due to foreground-background imbalance.

    The loss is formulated as a pixel-wise weighted sum of predicted probabilities and a
    signed distance map derived from the ground truth. The signed distance map is negative
    inside the foreground region and positive outside, with zero on the boundary.

    The data `input` (BNHW[D] where N is number of classes) is compared with ground truth `target`
    (BNHW[D]).
    Note that axis N of `input` is expected to be logits or probabilities for each class, if passing logits as input,
    must set `sigmoid=True` or `softmax=True`, or specifying `other_act`. And the same axis of `target`
    can be 1 or N (one-hot format).

    The original paper:
        Kervadec, H. et al. (2019) Boundary loss for highly unbalanced segmentation. MIDL 2019.
        https://arxiv.org/abs/1812.07032

    Example:
        >>> import torch
        >>> from monai.losses import BoundaryLoss
        >>> B, C, H, W = 2, 3, 5, 5
        >>> input = torch.rand(B, C, H, W)
        >>> target = torch.randint(0, C, size=(B, H, W))
        >>> bl = BoundaryLoss(softmax=True, to_onehot_y=True)
        >>> loss = bl(input, target)
    """

    def __init__(
        self,
        include_background: bool = True,
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
            batch: whether to compute the distance map and loss over the batch dimension before the dividing.
                Defaults to False, a boundary loss value is computed independently from each item in the batch
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
        self.batch = batch

    @torch.no_grad()
    def compute_distance_map(self, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the signed distance map for each class in the target.

        The signed distance map is negative inside the foreground region and positive outside,
        with zero on the boundary.

        Args:
            target: target tensor of shape BNHW[D], with values in {0, 1} (one-hot encoded).

        Returns:
            Signed distance map of the same shape as target.
        """
        if target.dim() not in (4, 5):
            raise ValueError("Only 2D (BNHW) and 3D (BNHWD) supported")

        distance_map = torch.zeros_like(target, dtype=torch.float32)

        for batch_idx in range(target.shape[0]):
            for channel_idx in range(target.shape[1]):
                mask = target[batch_idx, channel_idx : channel_idx + 1] > 0.5

                # Empty or full masks do not have a foreground/background interface.
                if not mask.any() or mask.all():
                    continue

                fg_dist: torch.Tensor = distance_transform_edt(mask)  # type: ignore
                bg_dist: torch.Tensor = distance_transform_edt(~mask)  # type: ignore

                signed = torch.zeros_like(mask, dtype=torch.float32)
                signed[mask] = -(fg_dist[mask].to(torch.float32) - 1)
                signed[~mask] = bg_dist[~mask].to(torch.float32)

                distance_map[batch_idx, channel_idx] = signed[0]

        return distance_map

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNHW[D], where N is the number of classes.
            target: the shape should be BNHW[D] or B1HW[D], where N is the number of classes.

        Raises:
            ValueError: If the input is not 2D (BNHW) or 3D (BNHWD).
            AssertionError: When input and target (after one hot transform if set)
                have different shapes.
            ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].

        Example:
            >>> import torch
            >>> from monai.losses import BoundaryLoss
            >>> B, C, H, W = 2, 3, 5, 5
            >>> input = torch.rand(B, C, H, W)
            >>> target_idx = torch.randint(0, C, size=(B, H, W)).long()
            >>> target = one_hot(target_idx[:, None, ...], num_classes=C)
            >>> bl = BoundaryLoss(softmax=True)
            >>> loss = bl(input, target)
        """
        if input.dim() not in (4, 5):
            raise ValueError("Only 2D (BNHW) and 3D (BNHWD) supported")

        n_pred_ch = input.shape[1]

        # Apply activation to input
        if self.sigmoid:
            input = torch.sigmoid(input)

        if self.softmax:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `softmax=True` ignored.", stacklevel=2)
            else:
                input = torch.softmax(input, dim=1)

        if self.other_act is not None:
            input = self.other_act(input)

        # Convert target to one-hot if needed
        if self.to_onehot_y:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `to_onehot_y=True` ignored.", stacklevel=2)
            else:
                if target.dim() == input.dim() - 1:
                    target = target.unsqueeze(dim=1)
                target = one_hot(target, num_classes=n_pred_ch)

        # Validate shapes match
        if input.shape != target.shape:
            raise AssertionError(f"input and target shapes do not match: {input.shape} vs {target.shape}")

        # Exclude background if requested
        if not self.include_background:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `include_background=False` ignored.", stacklevel=2)
            else:
                input = input[:, 1:]
                target = target[:, 1:]

        # Compute signed distance maps from target
        distance_map = self.compute_distance_map(target)

        # Compute boundary loss: sum over spatial dimensions of (probabilities * distance_map)
        # Then average over classes and batch
        spatial_axes = list(range(2, input.dim()))

        loss = torch.sum(input * distance_map, dim=spatial_axes)

        # Normalize by number of pixels per class per batch element
        num_pixels = torch.prod(torch.as_tensor(input.shape[2:], device=input.device))
        loss = loss / num_pixels
        if self.batch:
            loss = loss.mean(dim=0)

        if self.reduction == LossReduction.MEAN.value:
            loss = loss.mean()
        elif self.reduction == LossReduction.SUM.value:
            loss = loss.sum()
        elif self.reduction == LossReduction.NONE.value:
            # Return shape (B, C') unless batch=True reduces the batch dimension first.
            pass
        else:
            raise ValueError(f"Unsupported reduction: {self.reduction}")

        return loss

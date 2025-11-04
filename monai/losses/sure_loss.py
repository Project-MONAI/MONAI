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

from typing import Callable, Optional

import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss


def complex_diff_abs_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    First compute the difference in the complex domain,
    then get the absolute value and take the mse

    Args:
        x, y - B, 2, H, W real valued tensors representing complex numbers
                or  B,1,H,W complex valued tensors
    Returns:
        l2_loss - scalar
    """
    if not x.is_complex():
        x = torch.view_as_complex(x.permute(0, 2, 3, 1).contiguous())
    if not y.is_complex():
        y = torch.view_as_complex(y.permute(0, 2, 3, 1).contiguous())

    diff = torch.abs(x - y)
    return nn.functional.mse_loss(diff, torch.zeros_like(diff), reduction="mean")


def sure_loss_function(
    operator: Callable,
    x: torch.Tensor,
    y_pseudo_gt: torch.Tensor,
    y_ref: Optional[torch.Tensor] = None,
    eps: Optional[float] = -1.0,
    perturb_noise: Optional[torch.Tensor] = None,
    complex_input: Optional[bool] = False,
) -> torch.Tensor:
    """
    Args:
        operator (function): The operator function that takes in an input
        tensor x and returns an output tensor y. We will use this to compute
        the divergence. More specifically, we will perturb the input x by a
        small amount and compute the divergence between the perturbed output
        and the reference output

        x (torch.Tensor): The input tensor of shape (B, C, H, W) to the
        operator.  For complex input, the shape is (B, 2, H, W) aka C=2 real.
        For real input, the shape is (B, 1, H, W) real.

        y_pseudo_gt (torch.Tensor): The pseudo ground truth tensor of shape
        (B, C, H, W) used to compute the L2 loss.  For complex input, the shape is
        (B, 2, H, W) aka C=2 real.  For real input, the shape is (B, 1, H, W)
        real.

        y_ref (torch.Tensor, optional): The reference output tensor of shape
        (B, C, H, W) used to compute the divergence. Defaults to None.  For
        complex input, the shape is (B, 2, H, W) aka C=2 real.  For real input,
        the shape is (B, 1, H, W) real.

        eps (float, optional): The perturbation scalar. Set to -1 to set it
        automatically estimated based on y_pseudo_gtk

        perturb_noise (torch.Tensor, optional): The noise vector of shape (B, C, H, W).
        Defaults to None.  For complex input, the shape is (B, 2, H, W) aka C=2 real.
        For real input, the shape is (B, 1, H, W) real.

        complex_input(bool, optional): Whether the input is complex or not.
        Defaults to False.

    Returns:
        sure_loss (torch.Tensor): The SURE loss scalar.
    """
    # perturb input
    if perturb_noise is None:
        perturb_noise = torch.randn_like(x)
    if eps == -1.0:
        eps = float(torch.abs(y_pseudo_gt.max())) / 1000
    # get y_ref if not provided
    if y_ref is None:
        y_ref = operator(x)

    # get perturbed output
    x_perturbed = x + eps * perturb_noise  # type: ignore
    y_perturbed = operator(x_perturbed)
    # divergence
    divergence = torch.sum(1.0 / eps * torch.matmul(perturb_noise.permute(0, 1, 3, 2), y_perturbed - y_ref))  # type: ignore
    # l2 loss between y_ref, y_pseudo_gt
    if complex_input:
        l2_loss = complex_diff_abs_loss(y_ref, y_pseudo_gt)
    else:
        # real input
        l2_loss = nn.functional.mse_loss(y_ref, y_pseudo_gt, reduction="mean")

    # sure loss
    sure_loss = l2_loss * divergence / (x.shape[0] * x.shape[2] * x.shape[3])
    return sure_loss


class SURELoss(_Loss):
    """
    Calculate the Stein's Unbiased Risk Estimator (SURE) loss for a given operator.

    This is a differentiable loss function that can be used to train/guide an
    operator (e.g. neural network), where the pseudo ground truth is available
    but the reference ground truth is not. For example, in the MRI
    reconstruction, the pseudo ground truth is the zero-filled reconstruction
    and the reference ground truth is the fully sampled reconstruction.  Often,
    the reference ground truth is not available due to the lack of fully sampled
    data.

    The original SURE loss is proposed in [1]. The SURE loss used for guiding
    the diffusion model based MRI reconstruction is proposed in [2].

    Reference

    [1] Stein, C.M.: Estimation of the mean of a multivariate normal distribution. Annals of Statistics

    [2] B. Ozturkler et al. SMRD: SURE-based Robust MRI Reconstruction with Diffusion Models.
    (https://arxiv.org/pdf/2310.01799.pdf)
    """

    def __init__(self, perturb_noise: Optional[torch.Tensor] = None, eps: Optional[float] = None) -> None:
        """
        Args:
            perturb_noise (torch.Tensor, optional): The noise vector of shape
            (B, C, H, W). Defaults to None.  For complex input, the shape is (B, 2, H, W) aka C=2 real.
            For real input, the shape is (B, 1, H, W) real.

            eps (float, optional): The perturbation scalar. Defaults to None.
        """
        super().__init__()
        self.perturb_noise = perturb_noise
        self.eps = eps

    def forward(
        self,
        operator: Callable,
        x: torch.Tensor,
        y_pseudo_gt: torch.Tensor,
        y_ref: Optional[torch.Tensor] = None,
        complex_input: Optional[bool] = False,
    ) -> torch.Tensor:
        """
        Args:
            operator (function): The operator function that takes in an input
            tensor x and returns an output tensor y. We will use this to compute
            the divergence. More specifically, we will perturb the input x by a
            small amount and compute the divergence between the perturbed output
            and the reference output

            x (torch.Tensor): The input tensor of shape (B, C, H, W) to the
            operator. C=1 or 2: For complex input, the shape is (B, 2, H, W) aka
            C=2 real.  For real input, the shape is (B, 1, H, W) real.

            y_pseudo_gt (torch.Tensor): The pseudo ground truth tensor of shape
            (B, C, H, W) used to compute the L2 loss. C=1 or 2: For complex
            input, the shape is (B, 2, H, W) aka C=2 real.  For real input, the
            shape is (B, 1, H, W) real.

            y_ref (torch.Tensor, optional): The reference output tensor of the
            same shape as y_pseudo_gt

        Returns:
            sure_loss (torch.Tensor): The SURE loss scalar.
        """

        # check inputs shapes
        if x.dim() != 4:
            raise ValueError(f"Input tensor x should be 4D, got {x.dim()}.")
        if y_pseudo_gt.dim() != 4:
            raise ValueError(f"Input tensor y_pseudo_gt should be 4D, but got {y_pseudo_gt.dim()}.")
        if y_ref is not None and y_ref.dim() != 4:
            raise ValueError(f"Input tensor y_ref should be 4D, but got {y_ref.dim()}.")
        if x.shape != y_pseudo_gt.shape:
            raise ValueError(
                f"Input tensor x and y_pseudo_gt should have the same shape, but got x shape {x.shape}, "
                f"y_pseudo_gt shape {y_pseudo_gt.shape}."
            )
        if y_ref is not None and y_pseudo_gt.shape != y_ref.shape:
            raise ValueError(
                f"Input tensor y_pseudo_gt and y_ref should have the same shape, but got y_pseudo_gt shape {y_pseudo_gt.shape}, "
                f"y_ref shape {y_ref.shape}."
            )

        # compute loss
        loss = sure_loss_function(operator, x, y_pseudo_gt, y_ref, self.eps, self.perturb_noise, complex_input)

        return loss

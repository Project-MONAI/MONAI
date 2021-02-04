from typing import List, Optional

import torch
from torch.nn.modules.loss import _Loss

from monai.networks.layers import gaussian_1d, separable_filtering


def make_gaussian_kernel(sigma: int) -> torch.Tensor:
    if sigma <= 0:
        raise ValueError(f"expecting postive sigma, got sign={sigma}")
    sigma = torch.tensor(sigma)
    kernel = gaussian_1d(sigma=sigma, truncated=3, approx="sampled", normalize=False)
    return kernel


def make_cauchy_kernel(sigma: int) -> torch.Tensor:
    if sigma <= 0:
        raise ValueError(f"expecting postive sigma, got sign={sigma}")
    tail = int(sigma * 5)
    k = torch.tensor([((x / sigma) ** 2 + 1) for x in range(-tail, tail + 1)])
    k = torch.reciprocal(k)
    k = k / torch.sum(k)
    return k


kernel_fn_dict = {
    "gaussian": make_gaussian_kernel,
    "cauchy": make_cauchy_kernel,
}


class MultiScaleLoss(_Loss):
    """
    This is a wrapper class.
    It smooths the input and target at different scales before passing them into the wrapped loss function.
    The output is the average loss at all scales.

    Adapted from:
        DeepReg (https://github.com/DeepRegNet/DeepReg)
    """

    def __init__(
        self,
        loss: _Loss,
        scales: Optional[List] = None,
        kernel: str = "gaussian",
    ) -> None:
        """
        Args:
            loss: loss function to be wrapped
            scales: list of scalars or None, if None, do not apply any scaling.
            kernel: gaussian or cauchy.
        """
        super(MultiScaleLoss, self).__init__()
        if kernel not in kernel_fn_dict.keys():
            raise ValueError(f"got unsupported kernel type: {kernel}", "only support gaussian and cauchy")
        self.kernel_fn = kernel_fn_dict[kernel]
        self.loss = loss
        self.scales = scales

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        if self.scales is None:
            return self.loss(y_pred, y_true)
        losses = []
        for s in self.scales:
            if s == 0:
                # no smoothing
                losses.append(self.loss(y_pred, y_true))
            else:
                losses.append(
                    self.loss(
                        separable_filtering(y_pred, [self.kernel_fn(s)] * (y_true.ndim - 2)),
                        separable_filtering(y_true, [self.kernel_fn(s)] * (y_true.ndim - 2)),
                    )
                )
        loss = torch.mean(torch.stack(losses, dim=0), dim=0)
        return loss

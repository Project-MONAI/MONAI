from typing import Union

import torch
from torch.nn.modules.loss import _Loss

from monai.utils import LossReduction


def spatial_gradient(input: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Calculate gradients on single dimension of a tensor using central finite difference.
    It moves the tensor along the dimension to calculate the approximate gradient
    dx[i] = (x[i+1] - x[i-1]) / 2.
    Adapted from:
        DeepReg (https://github.com/DeepRegNet/DeepReg)

    Args:
        input: the shape should be BCH(WD).
        dim: dimension to calculate gradient along.
    Returns:
        gradient_dx: the shape should be BCH(WD)
    """
    slice_1 = slice(1, -1)
    slice_2_s = slice(2, None)
    slice_2_e = slice(None, -2)
    slice_all = slice(None)
    slicing_s, slicing_e = [slice_all, slice_all], [slice_all, slice_all]
    while len(slicing_s) < input.ndim:
        slicing_s = slicing_s + [slice_1]
        slicing_e = slicing_e + [slice_1]
    slicing_s[dim] = slice_2_s
    slicing_e[dim] = slice_2_e
    return (input[slicing_s] - input[slicing_e]) / 2.0


class BendingEnergyLoss(_Loss):
    """
    Calculate the bending energy based on second-order differentiation of input using central finite difference.

    Adapted from:
        DeepReg (https://github.com/DeepRegNet/DeepReg)
    """

    def __init__(
        self,
        reduction: Union[LossReduction, str] = LossReduction.MEAN,
    ) -> None:
        """
        Args:
            reduction: {``"none"``, ``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.

                - ``"none"``: no reduction will be applied.
                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.
        """
        super(BendingEnergyLoss, self).__init__(reduction=LossReduction(reduction).value)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BCH(WD)

        Raises:
            ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].

        """
        assert input.ndim in [3, 4, 5], f"expecting 3-d, 4-d or 5-d input, instead got input of shape {input.shape}"
        if input.ndim == 3:
            assert input.shape[-1] > 4, f"all spatial dimensions must > 4, got input of shape {input.shape}"
        elif input.ndim == 4:
            assert (
                input.shape[-1] > 4 and input.shape[-2] > 4
            ), f"all spatial dimensions must > 4, got input of shape {input.shape}"
        elif input.ndim == 5:
            assert (
                input.shape[-1] > 4 and input.shape[-2] > 4 and input.shape[-3] > 4
            ), f"all spatial dimensions must > 4, got input of shape {input.shape}"

        # first order gradient
        first_order_gradient = [spatial_gradient(input, dim) for dim in range(2, input.ndim)]

        energy = torch.tensor(0)
        for dim_1, g in enumerate(first_order_gradient):
            dim_1 += 2
            energy = spatial_gradient(g, dim_1) ** 2 + energy
            for dim_2 in range(dim_1 + 1, input.ndim):
                energy = 2 * spatial_gradient(g, dim_2) ** 2 + energy

        if self.reduction == LossReduction.MEAN.value:
            energy = torch.mean(energy)  # the batch and channel average
        elif self.reduction == LossReduction.SUM.value:
            energy = torch.sum(energy)  # sum over the batch and channel dims
        elif self.reduction == LossReduction.NONE.value:
            pass  # returns [N, n_classes] losses
        else:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')

        return energy

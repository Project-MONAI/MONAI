from typing import Union

import torch
from torch.nn.modules.loss import _Loss

from monai.utils import LossReduction


def gradient_dx(fx: torch.Tensor) -> torch.Tensor:
    """
    Calculate gradients on x-axis of a 3D tensor using central finite difference.
    It moves the tensor along axis 1 to calculate the approximate gradient, the x axis,
    dx[i] = (x[i+1] - x[i-1]) / 2.
    Args:
        fx: the shape should be BDHW.

    Returns:
        gradient_dx: the shape should be BDHW
    """
    return (fx[..., 1:-1, 1:-1, 2:] - fx[..., 1:-1, 1:-1, :-2]) / 2


def gradient_dy(fy: torch.Tensor) -> torch.Tensor:
    """
    Calculate gradients on y-axis of a 3D tensor using central finite difference.
    It moves the tensor along axis 1 to calculate the approximate gradient, the y axis,
    dy[i] = (y[i+1] - y[i-1]) / 2.
    Args:
        fy: the shape should be BDHW.

    Returns:
        gradient_dy: the shape should be BDHW
    """
    return (fy[..., 1:-1, 2:, 1:-1] - fy[..., 1:-1, :-2, 1:-1]) / 2


def gradient_dz(fz: torch.Tensor) -> torch.Tensor:
    """
    Calculate gradients on z-axis of a 3D tensor using central finite difference.
    It moves the tensor along axis 1 to calculate the approximate gradient, the z axis,
    dz[i] = (z[i+1] - z[i-1]) / 2.
    Args:
        fz: the shape should be BDHW.

    Returns:
        gradient_dy: the shape should be BDHW
    """
    return (fz[..., 2:, 1:-1, 1:-1] - fz[..., :-2, 1:-1, 1:-1]) / 2


class BendingEnergyLoss(_Loss):
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
        Calculate the bending energy based on second-order differentiation of input using central finite difference.

        Args:
            input: the shape should be B3DHW

        Raises:
            ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].

        """
        assert len(input.shape) == 5 and input.shape[1] == 3, (
            f"expecting 5-d ddf input with 3 channels, " f"instead got input of shape {input.shape}"
        )
        assert (
            input.shape[-1] > 4 and input.shape[-2] > 4 and input.shape[-3] > 4
        ), f"all depth, height and width must > 4, got input of shape {input.shape}"

        # first order gradient
        # (batch, 3, d-2, h-2, w-2)
        dfdx = gradient_dx(input)
        dfdy = gradient_dy(input)
        dfdz = gradient_dz(input)

        # second order gradient
        # (batch, 3, d-4, h-4, w-4)
        dfdxx = gradient_dx(dfdx)
        dfdyy = gradient_dy(dfdy)
        dfdzz = gradient_dz(dfdz)
        dfdxy = gradient_dy(dfdx)
        dfdyz = gradient_dz(dfdy)
        dfdxz = gradient_dz(dfdx)

        # (dx + dy + dz) ** 2 = dxx + dyy + dzz + 2*(dxy + dyz + dzx)
        energy = dfdxx ** 2 + dfdyy ** 2 + dfdzz ** 2
        energy += 2 * dfdxy ** 2 + 2 * dfdxz ** 2 + 2 * dfdyz ** 2

        if self.reduction == LossReduction.MEAN.value:
            energy = torch.mean(energy)  # the batch and channel average
        elif self.reduction == LossReduction.SUM.value:
            energy = torch.sum(energy)  # sum over the batch and channel dims
        elif self.reduction == LossReduction.NONE.value:
            pass  # returns [N, n_classes] losses
        else:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')

        return energy

import torch
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss

from monai.utils import LossReduction, Union

conv_dict = {1: F.conv1d, 2: F.conv2d, 3: F.conv3d}

EPS = 1e-7


class LocalNormalizedCrossCorrelationLoss(_Loss):
    """
    Local squared zero-normalized cross-correlation.
    The loss is based on a moving kernel/window over the y_true/y_pred,
    within the window the square of zncc is calculated.
    The kernel can be a rectangular / triangular / gaussian window.
    The final loss is the averaged loss over all windows.

    Adapted from:
        https://github.com/voxelmorph/voxelmorph/blob/legacy/src/losses.py
        DeepReg (https://github.com/DeepRegNet/DeepReg)
    """

    def __init__(
        self,
        in_channels: int,
        ndim: int = 3,
        kernel_size: int = 9,
        kernel_type: str = "rectangular",
        reduction: Union[LossReduction, str] = LossReduction.MEAN,
    ) -> None:
        """
        Args:
            in_channels: number of input channels
            ndim: number of spatial ndimensions, {``1``, ``2``, ``3``}. Defaults to 3.
            kernel_size: kernel size or kernel sigma for kernel_type=``"gaussian"``
            kernel_type: {``"rectangular"``, ``"triangular"``, ``"gaussian"``}. Defaults to ``"rectangular"``.
            reduction: {``"none"``, ``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.

                - ``"none"``: no reduction will be applied.
                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.
        """
        super(LocalNormalizedCrossCorrelation, self).__init__(reduction=LossReduction(reduction).value)
        self.in_channels = in_channels
        self.ndim = ndim
        if self.ndim not in [1, 2, 3]:
            raise ValueError(f"Unsupported ndim: {self.ndim}-d, only 1-d, 2-d, and 3-d inputs are supported")
        self.kernel_size = kernel_size
        if kernel_type == "rectangular":
            self.kernel, self.kernel_vol, self.padding = self.make_rectangular_kernel()
        elif kernel_type == "triangular":
            self.kernel, self.kernel_vol, self.padding = self.make_triangular_kernel()
        elif kernel_type == "gaussian":
            self.kernel, self.kernel_vol, self.padding = self.make_gaussian_kernel()
        else:
            raise ValueError(
                f'Unsupported kernel_type: {kernel_type}, available options are ["rectangular", "triangular", "gaussian"].'
            )

    def make_rectangular_kernel(self):
        shape = [1, self.in_channels] + [self.kernel_size] * self.ndim
        return torch.ones(shape, dtype=torch.float), self.kernel_size ** self.ndim, int((self.kernel_size - 1) / 2)

    def make_triangular_kernel(self):
        fsize = torch.tensor((self.kernel_size + 1) / 2, dtype=torch.int)
        f1 = torch.ones([1, 1] + [fsize] * self.ndim, dtype=torch.float) / fsize  # (1, 1, D, H, W)
        f2 = (
            torch.ones([self.in_channels, 1] + [fsize] * self.ndim, dtype=torch.float) / fsize
        )  # (1, in_channels, D, H, W)
        # (1, 1, D, H, W) -> (1, in_channels, D, H, W)
        fn = conv_dict[self.ndim]
        kernel = fn(f1, f2, padding=int((fsize - 1) / 2))

        return kernel, torch.sum(kernel ** 2), int((fsize - 1) / 2)

    def make_gaussian_kernel(self):
        mean = (self.kernel_size - 1) / 2.0
        sigma = self.kernel_size / 3

        grid_ndim = torch.arange(0, self.kernel_size)
        grid_ndim_ch = torch.arange(0, self.in_channels)

        if self.ndim == 1:
            grid = torch.meshgrid(grid_ndim_ch, grid_ndim)
        elif self.ndim == 2:
            grid = torch.meshgrid(grid_ndim_ch, grid_ndim, grid_ndim)
        elif self.ndim == 3:
            grid = torch.meshgrid(grid_ndim_ch, grid_ndim, grid_ndim, grid_ndim)
        else:
            raise ValueError

        grid = torch.stack(grid, dim=-1).to(dtype=torch.float)
        kernel = torch.exp(-torch.sum(torch.square(grid - mean), dim=-1) / (2 * sigma ** 2)).unsqueeze(
            0
        )  # (1, in_channel, kernel_size, kernel_size, kernel_size)
        return kernel, torch.sum(kernel ** 2), int((self.kernel_size - 1) / 2)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD].
            target: the shape should be BNH[WD].
        Raises:
            ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].
        """
        assert (
            input.shape[1] == self.in_channels
        ), f"expecting input with {self.in_channels} channels, got input of shape {input.shape}"
        assert (
            input.ndim - 2 == self.ndim
        ), f"expecting input with {self.ndim} spatial dimensions, got input of shape {input.shape}"
        assert (
            target.shape == input.shape
        ), f"ground truth has differing shape ({target.shape}) from input ({input.shape})"

        t2, p2, tp = target ** 2, input ** 2, target * input

        # sum over kernel
        fn = conv_dict[self.ndim]
        t_sum = fn(target, weight=self.kernel, padding=self.padding)
        p_sum = fn(input, weight=self.kernel, padding=self.padding)
        t2_sum = fn(t2, weight=self.kernel, padding=self.padding)
        p2_sum = fn(p2, weight=self.kernel, padding=self.padding)
        tp_sum = fn(tp, weight=self.kernel, padding=self.padding)

        # average over kernel
        t_avg = t_sum / self.kernel_vol
        p_avg = p_sum / self.kernel_vol

        # normalized cross correlation between t and p
        # sum[(t - mean[t]) * (p - mean[p])] / std[t] / std[p]
        # denoted by num / denom
        # assume we sum over N values
        # num = sum[t * p - mean[t] * p - t * mean[p] + mean[t] * mean[p]]
        #     = sum[t*p] - sum[t] * sum[p] / N * 2 + sum[t] * sum[p] / N
        #     = sum[t*p] - sum[t] * sum[p] / N
        #     = sum[t*p] - sum[t] * mean[p] = cross
        # the following is actually squared ncc
        cross = tp_sum - p_avg * t_sum
        t_var = t2_sum - t_avg * t_sum  # std[t] ** 2
        p_var = p2_sum - p_avg * p_sum  # std[p] ** 2
        ncc = (cross * cross + EPS) / (t_var * p_var + EPS)  # shape = (batch, 1, D, H, W)

        if self.reduction == LossReduction.SUM.value:
            return -torch.sum(ncc)  # sum over the batch and channel ndims
        if self.reduction == LossReduction.NONE.value:
            return -ncc
        if self.reduction == LossReduction.MEAN.value:
            return -torch.mean(ncc)  # average over the batch and channel ndims
        raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')

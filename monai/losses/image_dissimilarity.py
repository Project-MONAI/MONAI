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
from typing import Optional, Tuple, Union

import torch
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss

from monai.networks.layers import gaussian_1d, separable_filtering
from monai.utils import LossReduction, deprecated_arg
from monai.utils.module import look_up_option


def make_rectangular_kernel(kernel_size: int) -> torch.Tensor:
    return torch.ones(kernel_size)


def make_triangular_kernel(kernel_size: int) -> torch.Tensor:
    fsize = (kernel_size + 1) // 2
    if fsize % 2 == 0:
        fsize -= 1
    f = torch.ones((1, 1, fsize), dtype=torch.float).div(fsize)
    padding = (kernel_size - fsize) // 2 + fsize // 2
    return F.conv1d(f, f, padding=padding).reshape(-1)


def make_gaussian_kernel(kernel_size: int) -> torch.Tensor:
    sigma = torch.tensor(kernel_size / 3.0)
    kernel = gaussian_1d(sigma=sigma, truncated=kernel_size // 2, approx="sampled", normalize=False) * (
        2.5066282 * sigma
    )
    return kernel[:kernel_size]


kernel_dict = {
    "rectangular": make_rectangular_kernel,
    "triangular": make_triangular_kernel,
    "gaussian": make_gaussian_kernel,
}


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

    @deprecated_arg(name="ndim", since="0.6", msg_suffix="Please use `spatial_dims` instead.")
    def __init__(
        self,
        spatial_dims: int = 3,
        kernel_size: int = 3,
        kernel_type: str = "rectangular",
        reduction: Union[LossReduction, str] = LossReduction.MEAN,
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
        ndim: Optional[int] = None,
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions, {``1``, ``2``, ``3``}. Defaults to 3.
            kernel_size: kernel spatial size, must be odd.
            kernel_type: {``"rectangular"``, ``"triangular"``, ``"gaussian"``}. Defaults to ``"rectangular"``.
            reduction: {``"none"``, ``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.

                - ``"none"``: no reduction will be applied.
                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.
            smooth_nr: a small constant added to the numerator to avoid nan.
            smooth_dr: a small constant added to the denominator to avoid nan.

        .. deprecated:: 0.6.0
            ``ndim`` is deprecated, use ``spatial_dims``.
        """
        super().__init__(reduction=LossReduction(reduction).value)

        if ndim is not None:
            spatial_dims = ndim
        self.ndim = spatial_dims
        if self.ndim not in {1, 2, 3}:
            raise ValueError(f"Unsupported ndim: {self.ndim}-d, only 1-d, 2-d, and 3-d inputs are supported")

        self.kernel_size = kernel_size
        if self.kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, got {self.kernel_size}")

        _kernel = look_up_option(kernel_type, kernel_dict)
        self.kernel = _kernel(self.kernel_size)
        self.kernel_vol = self.get_kernel_vol()

        self.smooth_nr = float(smooth_nr)
        self.smooth_dr = float(smooth_dr)

    def get_kernel_vol(self):
        vol = self.kernel
        for _ in range(self.ndim - 1):
            vol = torch.matmul(vol.unsqueeze(-1), self.kernel.unsqueeze(0))
        return torch.sum(vol)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: the shape should be BNH[WD].
            target: the shape should be BNH[WD].
        Raises:
            ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].
        """
        if pred.ndim - 2 != self.ndim:
            raise ValueError(f"expecting pred with {self.ndim} spatial dimensions, got pred of shape {pred.shape}")
        if target.shape != pred.shape:
            raise ValueError(f"ground truth has differing shape ({target.shape}) from pred ({pred.shape})")

        t2, p2, tp = target**2, pred**2, target * pred
        kernel, kernel_vol = self.kernel.to(pred), self.kernel_vol.to(pred)
        # sum over kernel
        t_sum = separable_filtering(target, kernels=[kernel.to(pred)] * self.ndim)
        p_sum = separable_filtering(pred, kernels=[kernel.to(pred)] * self.ndim)
        t2_sum = separable_filtering(t2, kernels=[kernel.to(pred)] * self.ndim)
        p2_sum = separable_filtering(p2, kernels=[kernel.to(pred)] * self.ndim)
        tp_sum = separable_filtering(tp, kernels=[kernel.to(pred)] * self.ndim)

        # average over kernel
        t_avg = t_sum / kernel_vol
        p_avg = p_sum / kernel_vol

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
        t_var = torch.max(t_var, torch.zeros_like(t_var))
        p_var = torch.max(p_var, torch.zeros_like(p_var))
        ncc: torch.Tensor = (cross * cross + self.smooth_nr) / (t_var * p_var + self.smooth_dr)
        # shape = (batch, 1, D, H, W)

        if self.reduction == LossReduction.SUM.value:
            return torch.sum(ncc).neg()  # sum over the batch, channel and spatial ndims
        if self.reduction == LossReduction.NONE.value:
            return ncc.neg()
        if self.reduction == LossReduction.MEAN.value:
            return torch.mean(ncc).neg()  # average over the batch, channel and spatial ndims
        raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')


class GlobalMutualInformationLoss(_Loss):
    """
    Differentiable global mutual information loss via Parzen windowing method.

    Reference:
        https://dspace.mit.edu/handle/1721.1/123142, Section 3.1, equation 3.1-3.5, Algorithm 1
    """

    def __init__(
        self,
        kernel_type: str = "gaussian",
        num_bins: int = 23,
        sigma_ratio: float = 0.5,
        reduction: Union[LossReduction, str] = LossReduction.MEAN,
        smooth_nr: float = 1e-7,
        smooth_dr: float = 1e-7,
    ) -> None:
        """
        Args:
            kernel_type: {``"gaussian"``, ``"b-spline"``}
                ``"gaussian"``: adapted from DeepReg
                Reference: https://dspace.mit.edu/handle/1721.1/123142, Section 3.1, equation 3.1-3.5, Algorithm 1.
                ``"b-spline"``: based on the method of Mattes et al [1,2] and adapted from ITK
                References:
                  [1] "Nonrigid multimodality image registration"
                      D. Mattes, D. R. Haynor, H. Vesselle, T. Lewellen and W. Eubank
                      Medical Imaging 2001: Image Processing, 2001, pp. 1609-1620.
                  [2] "PET-CT Image Registration in the Chest Using Free-form Deformations"
                      D. Mattes, D. R. Haynor, H. Vesselle, T. Lewellen and W. Eubank
                      IEEE Transactions in Medical Imaging. Vol.22, No.1,
                      January 2003. pp.120-128.

            num_bins: number of bins for intensity
            sigma_ratio: a hyper param for gaussian function
            reduction: {``"none"``, ``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.

                - ``"none"``: no reduction will be applied.
                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.
            smooth_nr: a small constant added to the numerator to avoid nan.
            smooth_dr: a small constant added to the denominator to avoid nan.
        """
        super().__init__(reduction=LossReduction(reduction).value)
        if num_bins <= 0:
            raise ValueError("num_bins must > 0, got {num_bins}")
        bin_centers = torch.linspace(0.0, 1.0, num_bins)  # (num_bins,)
        sigma = torch.mean(bin_centers[1:] - bin_centers[:-1]) * sigma_ratio
        self.kernel_type = look_up_option(kernel_type, ["gaussian", "b-spline"])
        self.num_bins = num_bins
        self.kernel_type = kernel_type
        if self.kernel_type == "gaussian":
            self.preterm = 1 / (2 * sigma**2)
            self.bin_centers = bin_centers[None, None, ...]
        self.smooth_nr = float(smooth_nr)
        self.smooth_dr = float(smooth_dr)

    def parzen_windowing(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.kernel_type == "gaussian":
            pred_weight, pred_probability = self.parzen_windowing_gaussian(pred)
            target_weight, target_probability = self.parzen_windowing_gaussian(target)
        elif self.kernel_type == "b-spline":
            # a third order BSpline kernel is used for the pred image intensity PDF.
            pred_weight, pred_probability = self.parzen_windowing_b_spline(pred, order=3)
            # a zero order (box car) BSpline kernel is used for the target image intensity PDF.
            target_weight, target_probability = self.parzen_windowing_b_spline(target, order=0)
        else:
            raise ValueError
        return pred_weight, pred_probability, target_weight, target_probability

    def parzen_windowing_b_spline(self, img: torch.Tensor, order: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parzen windowing with b-spline kernel (adapted from ITK)

        Args:
            img: the shape should be B[NDHW].
            order: int.
        """

        # Compute binsize for the histograms.
        #
        # The binsize for the image intensities needs to be adjusted so that
        # we can avoid dealing with boundary conditions using the cubic
        # spline as the Parzen window.  We do this by increasing the size
        # of the bins so that the joint histogram becomes "padded" at the
        # borders. Because we are changing the binsize,
        # we also need to shift the minimum by the padded amount in order to
        # avoid minimum values filling in our padded region.
        #
        # Note that there can still be non-zero bin values in the padded region,
        # it's just that these bins will never be a central bin for the Parzen
        # window.
        _max, _min = torch.max(img), torch.min(img)
        padding = 2
        bin_size = (_max - _min) / (self.num_bins - 2 * padding)
        norm_min = torch.div(_min, bin_size) - padding

        # assign bin/window index to each voxel
        window_term = torch.div(img, bin_size) - norm_min  # B[NDHW]
        # make sure the extreme values are in valid (non-padded) bins
        window_term = torch.clamp(window_term, padding, self.num_bins - padding - 1)  # B[NDHW]
        window_term = window_term.reshape(window_term.shape[0], -1, 1)  # (batch, num_sample, 1)
        bins = torch.arange(self.num_bins, device=window_term.device).reshape(1, 1, -1)  # (1, 1, num_bins)
        sample_bin_matrix = torch.abs(bins - window_term)  # (batch, num_sample, num_bins)

        # b-spleen kernel
        # (4 - 6 * abs ** 2 + 3 * abs ** 3) / 6 when 0 <= abs < 1
        # (2 - abs) ** 3 / 6 when 1 <= abs < 2
        weight = torch.zeros_like(sample_bin_matrix, dtype=torch.float)  # (batch, num_sample, num_bins)
        if order == 0:
            weight = weight + (sample_bin_matrix < 0.5) + (sample_bin_matrix == 0.5) * 0.5
        elif order == 3:
            weight = (
                weight + (4 - 6 * sample_bin_matrix**2 + 3 * sample_bin_matrix**3) * (sample_bin_matrix < 1) / 6
            )
            weight = weight + (2 - sample_bin_matrix) ** 3 * (sample_bin_matrix >= 1) * (sample_bin_matrix < 2) / 6
        else:
            raise ValueError(f"Do not support b-spline {order}-order parzen windowing")

        weight = weight / torch.sum(weight, dim=-1, keepdim=True)  # (batch, num_sample, num_bins)
        probability = torch.mean(weight, dim=-2, keepdim=True)  # (batch, 1, num_bins)
        return weight, probability

    def parzen_windowing_gaussian(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parzen windowing with gaussian kernel (adapted from DeepReg implementation)
        Note: the input is expected to range between 0 and 1
        Args:
            img: the shape should be B[NDHW].
        """
        img = torch.clamp(img, 0, 1)
        img = img.reshape(img.shape[0], -1, 1)  # (batch, num_sample, 1)
        weight = torch.exp(
            -self.preterm.to(img) * (img - self.bin_centers.to(img)) ** 2
        )  # (batch, num_sample, num_bin)
        weight = weight / torch.sum(weight, dim=-1, keepdim=True)  # (batch, num_sample, num_bin)
        probability = torch.mean(weight, dim=-2, keepdim=True)  # (batch, 1, num_bin)
        return weight, probability

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: the shape should be B[NDHW].
            target: the shape should be same as the pred shape.
        Raises:
            ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].
        """
        if target.shape != pred.shape:
            raise ValueError(f"ground truth has differing shape ({target.shape}) from pred ({pred.shape})")
        wa, pa, wb, pb = self.parzen_windowing(pred, target)  # (batch, num_sample, num_bin), (batch, 1, num_bin)

        pab = torch.bmm(wa.permute(0, 2, 1), wb.to(wa)).div(wa.shape[1])  # (batch, num_bins, num_bins)
        papb = torch.bmm(pa.permute(0, 2, 1), pb.to(pa))  # (batch, num_bins, num_bins)
        mi = torch.sum(
            pab * torch.log((pab + self.smooth_nr) / (papb + self.smooth_dr) + self.smooth_dr), dim=(1, 2)
        )  # (batch)

        if self.reduction == LossReduction.SUM.value:
            return torch.sum(mi).neg()  # sum over the batch and channel ndims
        if self.reduction == LossReduction.NONE.value:
            return mi.neg()
        if self.reduction == LossReduction.MEAN.value:
            return torch.mean(mi).neg()  # average over the batch and channel ndims
        raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')

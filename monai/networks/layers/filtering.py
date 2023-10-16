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

import torch

from monai.utils.module import optional_import

_C, _ = optional_import("monai._C")

__all__ = ["BilateralFilter", "PHLFilter", "TrainableBilateralFilter", "TrainableJointBilateralFilter"]


class BilateralFilter(torch.autograd.Function):
    """
    Blurs the input tensor spatially whilst preserving edges. Can run on 1D, 2D, or 3D,
    tensors (on top of Batch and Channel dimensions). Two implementations are provided,
    an exact solution and a much faster approximation which uses a permutohedral lattice.

    See:
        https://en.wikipedia.org/wiki/Bilateral_filter
        https://graphics.stanford.edu/papers/permutohedral/

    Args:
        input: input tensor.
        spatial_sigma: the standard deviation of the spatial blur. Higher values can
            hurt performance when not using the approximate method (see fast approx).
        color_sigma: the standard deviation of the color blur. Lower values preserve
            edges better whilst higher values tend to a simple gaussian spatial blur.
        fast approx: This flag chooses between two implementations. The approximate method may
            produce artifacts in some scenarios whereas the exact solution may be intolerably
            slow for high spatial standard deviations.

    Returns:
        output (torch.Tensor): output tensor.
    """

    @staticmethod
    def forward(ctx, input, spatial_sigma=5, color_sigma=0.5, fast_approx=True):
        """autograd forward"""
        ctx.ss = spatial_sigma
        ctx.cs = color_sigma
        ctx.fa = fast_approx
        output_data = _C.bilateral_filter(input, spatial_sigma, color_sigma, fast_approx)
        return output_data

    @staticmethod
    def backward(ctx, grad_output):
        """autograd backward"""
        spatial_sigma, color_sigma, fast_approx = ctx.ss, ctx.cs, ctx.fa
        grad_input = _C.bilateral_filter(grad_output, spatial_sigma, color_sigma, fast_approx)
        return grad_input, None, None, None


class PHLFilter(torch.autograd.Function):
    """
    Filters input based on arbitrary feature vectors. Uses a permutohedral
    lattice data structure to efficiently approximate n-dimensional gaussian
    filtering. Complexity is broadly independent of kernel size. Most applicable
    to higher filter dimensions and larger kernel sizes.

    See:
        https://graphics.stanford.edu/papers/permutohedral/

    Args:
        input: input tensor to be filtered.
        features: feature tensor used to filter the input.
        sigmas: the standard deviations of each feature in the filter.

    Returns:
        output (torch.Tensor): output tensor.
    """

    @staticmethod
    def forward(ctx, input, features, sigmas=None):
        scaled_features = features
        if sigmas is not None:
            for i in range(features.size(1)):
                scaled_features[:, i, ...] /= sigmas[i]

        ctx.save_for_backward(scaled_features)
        output_data = _C.phl_filter(input, scaled_features)
        return output_data

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("PHLFilter does not currently support Backpropagation")
        # scaled_features, = ctx.saved_variables
        # grad_input = _C.phl_filter(grad_output, scaled_features)
        # return grad_input


class TrainableBilateralFilterFunction(torch.autograd.Function):
    """
    torch.autograd.Function for the TrainableBilateralFilter layer.

    See:
        F. Wagner, et al., Ultralow-parameter denoising: Trainable bilateral filter layers in
        computed tomography, Medical Physics (2022), https://doi.org/10.1002/mp.15718

    Args:
        input: input tensor to be filtered.
        sigma x: trainable standard deviation of the spatial filter kernel in x direction.
        sigma y: trainable standard deviation of the spatial filter kernel in y direction.
        sigma z: trainable standard deviation of the spatial filter kernel in z direction.
        color sigma: trainable standard deviation of the intensity range kernel. This filter
            parameter determines the degree of edge preservation.

    Returns:
        output (torch.Tensor): filtered tensor.
    """

    @staticmethod
    def forward(ctx, input_img, sigma_x, sigma_y, sigma_z, color_sigma):
        output_tensor, output_weights_tensor, do_dx_ki, do_dsig_r, do_dsig_x, do_dsig_y, do_dsig_z = _C.tbf_forward(
            input_img, sigma_x, sigma_y, sigma_z, color_sigma
        )

        ctx.save_for_backward(
            input_img,
            sigma_x,
            sigma_y,
            sigma_z,
            color_sigma,
            output_tensor,
            output_weights_tensor,
            do_dx_ki,
            do_dsig_r,
            do_dsig_x,
            do_dsig_y,
            do_dsig_z,
        )

        return output_tensor

    @staticmethod
    def backward(ctx, grad_output):
        input_img = ctx.saved_tensors[0]  # input image
        sigma_x = ctx.saved_tensors[1]
        sigma_y = ctx.saved_tensors[2]
        sigma_z = ctx.saved_tensors[3]
        color_sigma = ctx.saved_tensors[4]
        output_tensor = ctx.saved_tensors[5]  # filtered image
        output_weights_tensor = ctx.saved_tensors[6]  # weights
        do_dx_ki = ctx.saved_tensors[7]  # derivative of output with respect to input, while k==i
        do_dsig_r = ctx.saved_tensors[8]  # derivative of output with respect to range sigma
        do_dsig_x = ctx.saved_tensors[9]  # derivative of output with respect to sigma x
        do_dsig_y = ctx.saved_tensors[10]  # derivative of output with respect to sigma y
        do_dsig_z = ctx.saved_tensors[11]  # derivative of output with respect to sigma z

        # calculate gradient with respect to the sigmas
        grad_color_sigma = torch.sum(grad_output * do_dsig_r)
        grad_sig_x = torch.sum(grad_output * do_dsig_x)
        grad_sig_y = torch.sum(grad_output * do_dsig_y)
        grad_sig_z = torch.sum(grad_output * do_dsig_z)

        grad_output_tensor = _C.tbf_backward(
            grad_output,
            input_img,
            output_tensor,
            output_weights_tensor,
            do_dx_ki,
            sigma_x,
            sigma_y,
            sigma_z,
            color_sigma,
        )

        return grad_output_tensor, grad_sig_x, grad_sig_y, grad_sig_z, grad_color_sigma


class TrainableBilateralFilter(torch.nn.Module):
    """
    Implementation of a trainable bilateral filter layer as proposed in the corresponding publication.
    All filter parameters can be trained data-driven. The spatial filter kernels x, y, and z determine
    image smoothing whereas the color parameter specifies the amount of edge preservation.
    Can run on 1D, 2D, or 3D tensors (on top of Batch and Channel dimensions).

    See:
        F. Wagner, et al., Ultralow-parameter denoising: Trainable bilateral filter layers in
        computed tomography, Medical Physics (2022), https://doi.org/10.1002/mp.15718

    Args:
        input: input tensor to be filtered.
        spatial_sigma: tuple (sigma_x, sigma_y, sigma_z) initializing the trainable standard
            deviations of the spatial filter kernels. Tuple length must equal the number of
            spatial input dimensions.
        color_sigma: trainable standard deviation of the intensity range kernel. This filter
            parameter determines the degree of edge preservation.

    Returns:
        output (torch.Tensor): filtered tensor.
    """

    def __init__(self, spatial_sigma, color_sigma):
        super().__init__()

        if isinstance(spatial_sigma, float):
            spatial_sigma = [spatial_sigma, spatial_sigma, spatial_sigma]
            self.len_spatial_sigma = 3
        elif len(spatial_sigma) == 1:
            spatial_sigma = [spatial_sigma[0], 0.01, 0.01]
            self.len_spatial_sigma = 1
        elif len(spatial_sigma) == 2:
            spatial_sigma = [spatial_sigma[0], spatial_sigma[1], 0.01]
            self.len_spatial_sigma = 2
        elif len(spatial_sigma) == 3:
            spatial_sigma = [spatial_sigma[0], spatial_sigma[1], spatial_sigma[2]]
            self.len_spatial_sigma = 3
        else:
            raise ValueError(
                f"len(spatial_sigma) {spatial_sigma} must match number of spatial dims {self.ken_spatial_sigma}."
            )

        # Register sigmas as trainable parameters.
        self.sigma_x = torch.nn.Parameter(torch.tensor(spatial_sigma[0]))
        self.sigma_y = torch.nn.Parameter(torch.tensor(spatial_sigma[1]))
        self.sigma_z = torch.nn.Parameter(torch.tensor(spatial_sigma[2]))
        self.sigma_color = torch.nn.Parameter(torch.tensor(color_sigma))

    def forward(self, input_tensor):
        if input_tensor.shape[1] != 1:
            raise ValueError(
                f"Currently channel dimensions >1 ({input_tensor.shape[1]}) are not supported. "
                "Please use multiple parallel filter layers if you want "
                "to filter multiple channels."
            )

        len_input = len(input_tensor.shape)

        # C++ extension so far only supports 5-dim inputs.
        if len_input == 3:
            input_tensor = input_tensor.unsqueeze(3).unsqueeze(4)
        elif len_input == 4:
            input_tensor = input_tensor.unsqueeze(4)

        if self.len_spatial_sigma != len_input:
            raise ValueError(f"Spatial dimension ({len_input}) must match initialized len(spatial_sigma).")

        prediction = TrainableBilateralFilterFunction.apply(
            input_tensor, self.sigma_x, self.sigma_y, self.sigma_z, self.sigma_color
        )

        # Make sure to return tensor of the same shape as the input.
        if len_input == 3:
            prediction = prediction.squeeze(4).squeeze(3)
        elif len_input == 4:
            prediction = prediction.squeeze(4)

        return prediction


class TrainableJointBilateralFilterFunction(torch.autograd.Function):
    """
    torch.autograd.Function for the TrainableJointBilateralFilter layer.

    See:
        F. Wagner, et al., Trainable joint bilateral filters for enhanced prediction stability in
        low-dose CT, Scientific Reports (2022), https://doi.org/10.1038/s41598-022-22530-4

    Args:
        input: input tensor to be filtered.
        guide: guidance image tensor to be used during filtering.
        sigma x: trainable standard deviation of the spatial filter kernel in x direction.
        sigma y: trainable standard deviation of the spatial filter kernel in y direction.
        sigma z: trainable standard deviation of the spatial filter kernel in z direction.
        color sigma: trainable standard deviation of the intensity range kernel. This filter
            parameter determines the degree of edge preservation.

    Returns:
        output (torch.Tensor): filtered tensor.
    """

    @staticmethod
    def forward(ctx, input_img, guidance_img, sigma_x, sigma_y, sigma_z, color_sigma):
        output_tensor, output_weights_tensor, do_dx_ki, do_dsig_r, do_dsig_x, do_dsig_y, do_dsig_z = _C.tjbf_forward(
            input_img, guidance_img, sigma_x, sigma_y, sigma_z, color_sigma
        )

        ctx.save_for_backward(
            input_img,
            sigma_x,
            sigma_y,
            sigma_z,
            color_sigma,
            output_tensor,
            output_weights_tensor,
            do_dx_ki,
            do_dsig_r,
            do_dsig_x,
            do_dsig_y,
            do_dsig_z,
            guidance_img,
        )

        return output_tensor

    @staticmethod
    def backward(ctx, grad_output):
        input_img = ctx.saved_tensors[0]  # input image
        sigma_x = ctx.saved_tensors[1]
        sigma_y = ctx.saved_tensors[2]
        sigma_z = ctx.saved_tensors[3]
        color_sigma = ctx.saved_tensors[4]
        output_tensor = ctx.saved_tensors[5]  # filtered image
        output_weights_tensor = ctx.saved_tensors[6]  # weights
        do_dx_ki = ctx.saved_tensors[7]  # derivative of output with respect to input, while k==i
        do_dsig_r = ctx.saved_tensors[8]  # derivative of output with respect to range sigma
        do_dsig_x = ctx.saved_tensors[9]  # derivative of output with respect to sigma x
        do_dsig_y = ctx.saved_tensors[10]  # derivative of output with respect to sigma y
        do_dsig_z = ctx.saved_tensors[11]  # derivative of output with respect to sigma z
        guidance_img = ctx.saved_tensors[12]  # guidance image

        # calculate gradient with respect to the sigmas
        grad_color_sigma = torch.sum(grad_output * do_dsig_r)
        grad_sig_x = torch.sum(grad_output * do_dsig_x)
        grad_sig_y = torch.sum(grad_output * do_dsig_y)
        grad_sig_z = torch.sum(grad_output * do_dsig_z)

        grad_output_tensor, grad_guidance_tensor = _C.tjbf_backward(
            grad_output,
            input_img,
            guidance_img,
            output_tensor,
            output_weights_tensor,
            do_dx_ki,
            sigma_x,
            sigma_y,
            sigma_z,
            color_sigma,
        )

        return grad_output_tensor, grad_guidance_tensor, grad_sig_x, grad_sig_y, grad_sig_z, grad_color_sigma


class TrainableJointBilateralFilter(torch.nn.Module):
    """
    Implementation of a trainable joint bilateral filter layer as proposed in the corresponding publication.
    The guidance image is used as additional (edge) information during filtering. All filter parameters and the
    guidance image can be trained data-driven. The spatial filter kernels x, y, and z determine
    image smoothing whereas the color parameter specifies the amount of edge preservation.
    Can run on 1D, 2D, or 3D tensors (on top of Batch and Channel dimensions). Input tensor shape must match
    guidance tensor shape.

    See:
        F. Wagner, et al., Trainable joint bilateral filters for enhanced prediction stability in
        low-dose CT, Scientific Reports (2022), https://doi.org/10.1038/s41598-022-22530-4

    Args:
        input: input tensor to be filtered.
        guide: guidance image tensor to be used during filtering.
        spatial_sigma: tuple (sigma_x, sigma_y, sigma_z) initializing the trainable standard
            deviations of the spatial filter kernels. Tuple length must equal the number of
            spatial input dimensions.
        color_sigma: trainable standard deviation of the intensity range kernel. This filter
            parameter determines the degree of edge preservation.

    Returns:
        output (torch.Tensor): filtered tensor.
    """

    def __init__(self, spatial_sigma, color_sigma):
        super().__init__()

        if isinstance(spatial_sigma, float):
            spatial_sigma = [spatial_sigma, spatial_sigma, spatial_sigma]
            self.len_spatial_sigma = 3
        elif len(spatial_sigma) == 1:
            spatial_sigma = [spatial_sigma[0], 0.01, 0.01]
            self.len_spatial_sigma = 1
        elif len(spatial_sigma) == 2:
            spatial_sigma = [spatial_sigma[0], spatial_sigma[1], 0.01]
            self.len_spatial_sigma = 2
        elif len(spatial_sigma) == 3:
            spatial_sigma = [spatial_sigma[0], spatial_sigma[1], spatial_sigma[2]]
            self.len_spatial_sigma = 3
        else:
            raise ValueError(
                f"len(spatial_sigma) {spatial_sigma} must match number of spatial dims {self.ken_spatial_sigma}."
            )

        # Register sigmas as trainable parameters.
        self.sigma_x = torch.nn.Parameter(torch.tensor(spatial_sigma[0]))
        self.sigma_y = torch.nn.Parameter(torch.tensor(spatial_sigma[1]))
        self.sigma_z = torch.nn.Parameter(torch.tensor(spatial_sigma[2]))
        self.sigma_color = torch.nn.Parameter(torch.tensor(color_sigma))

    def forward(self, input_tensor, guidance_tensor):
        if input_tensor.shape[1] != 1:
            raise ValueError(
                f"Currently channel dimensions >1 ({input_tensor.shape[1]}) are not supported. "
                "Please use multiple parallel filter layers if you want "
                "to filter multiple channels."
            )
        if input_tensor.shape != guidance_tensor.shape:
            raise ValueError(
                "Shape of input image must equal shape of guidance image."
                f"Got {input_tensor.shape} and {guidance_tensor.shape}."
            )

        len_input = len(input_tensor.shape)

        # C++ extension so far only supports 5-dim inputs.
        if len_input == 3:
            input_tensor = input_tensor.unsqueeze(3).unsqueeze(4)
            guidance_tensor = guidance_tensor.unsqueeze(3).unsqueeze(4)
        elif len_input == 4:
            input_tensor = input_tensor.unsqueeze(4)
            guidance_tensor = guidance_tensor.unsqueeze(4)

        if self.len_spatial_sigma != len_input:
            raise ValueError(f"Spatial dimension ({len_input}) must match initialized len(spatial_sigma).")

        prediction = TrainableJointBilateralFilterFunction.apply(
            input_tensor, guidance_tensor, self.sigma_x, self.sigma_y, self.sigma_z, self.sigma_color
        )

        # Make sure to return tensor of the same shape as the input.
        if len_input == 3:
            prediction = prediction.squeeze(4).squeeze(3)
        elif len_input == 4:
            prediction = prediction.squeeze(4)

        return prediction

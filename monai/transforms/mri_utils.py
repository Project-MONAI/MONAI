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
import numpy as np
import torch


def to_tensor(data):
    """
    Convert numpy array to PyTorch tensor. For complex arrays, the real and imaginary parts
    are stacked along the last dimension.
    inputs:
        data (np.array): Input numpy array
    outputs:
        torch.Tensor: PyTorch version of data
    """
    if np.iscomplexobj(data):
        data = np.stack((data.real, data.imag), axis=-1)
    return torch.from_numpy(data)


def complex_abs(x):
    """
    Compute the absolute value of a complex array.
    inputs:
        x (np.array): Input numpy array with 2 channels in the last
        dimension representing real and imaginary parts.
    outputs:
        np.array: Absolute value along the last dimention
    """
    assert x.shape[-1] == 2
    return np.sqrt(x[..., 0] ** 2 + x[..., 1] ** 2)


def roll(x, shift, dim):
    """
    Similar to np.roll but applies to PyTorch Tensors
    """
    if isinstance(shift, (tuple, list)):
        assert len(shift) == len(dim)
        for s, d in zip(shift, dim):
            x = roll(x, s, d)
        return x
    shift = shift % x.size(dim)
    if shift == 0:
        return x
    left = x.narrow(dim, 0, x.size(dim) - shift)
    right = x.narrow(dim, x.size(dim) - shift, shift)
    return torch.cat((right, left), dim=dim)


def fftshift(x, dim=None):
    """
    Similar to np.fft.fftshift but applies to PyTorch Tensors
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [dim // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = x.shape[dim] // 2
    else:
        shift = [x.shape[i] // 2 for i in dim]
    return roll(x, shift, dim)


def ifftshift(x, dim=None):
    """
    Similar to np.fft.ifftshift but applies to PyTorch Tensors
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [(dim + 1) // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = (x.shape[dim] + 1) // 2
    else:
        shift = [(x.shape[i] + 1) // 2 for i in dim]
    return roll(x, shift, dim)


def fft(input, signal_ndim, normalized=False):
    """
    This function is called from the fft2 function below
    """
    if signal_ndim < 1 or signal_ndim > 3:
        print("Signal ndim out of range, was", signal_ndim, "but expected a value between 1 and 3, inclusive")
        return

    dims = -1
    if signal_ndim == 2:
        dims = (-2, -1)
    if signal_ndim == 3:
        dims = (-3, -2, -1)

    norm = "backward"
    if normalized:
        norm = "ortho"

    return torch.view_as_real(torch.fft.fftn(torch.view_as_complex(input), dim=dims, norm=norm))


def ifft(input, signal_ndim, normalized=False):
    """
    This function is called from the ifft2 function below
    """
    if signal_ndim < 1 or signal_ndim > 3:
        print("Signal ndim out of range, was", signal_ndim, "but expected a value between 1 and 3, inclusive")
        return

    dims = -1
    if signal_ndim == 2:
        dims = (-2, -1)
    if signal_ndim == 3:
        dims = (-3, -2, -1)

    norm = "backward"
    if normalized:
        norm = "ortho"

    return torch.view_as_real(torch.fft.ifftn(torch.view_as_complex(input), dim=dims, norm=norm))


def fft2(data):
    """
    ref: https://github.com/facebookresearch/fastMRI/tree/master/fastmri
    Apply centered 2 dimensional Fast Fourier Transform. It calls the fft function above to make it compatible with the latest version of pytorch.
    inputs:
        data (torch.Tensor): Complex valued input data containing at least 3 dimensions: dimensions
            -3 & -2 are spatial dimensions and dimension -1 has size 2. All other dimensions are
            assumed to be batch dimensions.
    outputs:
        torch.Tensor: The FFT of the input.
    """
    assert data.size(-1) == 2
    data = ifftshift(data, dim=(-3, -2))
    data = fft(data, 2, normalized=True)
    data = fftshift(data, dim=(-3, -2))
    return data


def ifft2(data):
    """
    ref: https://github.com/facebookresearch/fastMRI/tree/master/fastmri
    Apply centered 2-dimensional Inverse Fast Fourier Transform. It calls the ifft function above to make it compatible with the latest version of pytorch.
    inputs:
        data (torch.Tensor): Complex valued input data containing at least 3 dimensions: dimensions
            -3 & -2 are spatial dimensions and dimension -1 has size 2. All other dimensions are
            assumed to be batch dimensions.
    outputs:
        torch.Tensor: The IFFT of the input.
    """
    assert data.size(-1) == 2
    data = ifftshift(data, dim=(-3, -2))
    data = ifft(data, 2, normalized=True)
    data = fftshift(data, dim=(-3, -2))
    return data


def apply_mask(data, mask_func=None, mask=None, seed=None):
    """
    Subsample given k-space by multiplying with a mask.
    inputs:
        data (torch.Tensor): The input k-space data. This should have at least 3 dimensions, where
            dimensions -3 and -2 are the spatial dimensions, and the final dimension has size
            2 (for complex values).
        mask_func (callable): A function that takes a shape (tuple of ints) and a random
            number seed and returns a mask.
        seed (int or 1-d array_like, optional): Seed for the random number generator.
    outputs:
        (tuple): tuple containing:
            masked data (torch.Tensor): Subsampled k-space data
            mask (torch.Tensor): The generated mask
    """
    shape = np.array(data.shape)
    shape[:-3] = 1
    if mask is None:
        mask = mask_func(shape, seed)
    return data * mask, mask


def center_crop(data, shape):
    """
    Apply a center crop to the input real image or batch of real images.
    inputs:
        data (torch.Tensor): The input tensor to be center cropped. It should have at
            least 2 dimensions and the cropping is applied along the last two dimensions.
        shape (int, int): The output shape. The shape should be smaller than the
            corresponding dimensions of data.
    outputs:
        torch.Tensor: The center cropped image
    """
    assert 0 < shape[0] <= data.shape[-2]
    assert 0 < shape[1] <= data.shape[-1]
    w_from = (data.shape[-2] - shape[0]) // 2
    h_from = (data.shape[-1] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]
    return data[..., w_from:w_to, h_from:h_to]


def normalize(data, mean, stddev, eps=0.0):
    """
    Normalize (standardize in this case) the given tensor using:
        (data - mean) / (stddev + eps)
    inputss:
        data (torch.Tensor): Input data to be normalized
        mean (float): Mean value
        stddev (float): Standard deviation
        eps (float): Added to stddev to prevent dividing by zero
    outputs:
        torch.Tensor: Normalized tensor
    """
    return (data - mean) / (stddev + eps)


def normalize_instance(data, eps=0.0):
    """
    Normalize (standardize in this case) the given tensor using:
        (data - mean) / (stddev + eps)
    where mean and stddev are computed from the data itself.
    inputs:
        data (torch.Tensor): Input data to be normalized
        eps (float): Added to stddev to prevent dividing by zero
    outputs:
        torch.Tensor: Normalized tensor
    """
    mean = data.mean()
    std = data.std()
    return normalize(data, mean, std, eps), mean, std

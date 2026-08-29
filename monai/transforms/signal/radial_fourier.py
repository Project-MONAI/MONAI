# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
3D Radial Fourier Transform for medical imaging data.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Optional, Union, cast

import torch
from torch.fft import fftn, fftshift, ifftn, ifftshift

from monai.config import NdarrayOrTensor
from monai.transforms.transform import Transform
from monai.utils import convert_data_type


class RadialFourier3D(Transform):
    """
    Computes the 3D Radial Fourier Transform of medical imaging data.

    This transform converts 3D medical images into radial frequency domain representations,
    which is particularly useful for handling anisotropic resolution common in medical scans
    (e.g., different resolution in axial vs coronal planes).
    The radial transform provides rotation-invariant frequency analysis and can help
    normalize frequency representations across datasets with different acquisition parameters.

    Args:
        normalize: If ``True``, normalize the output by the number of voxels.
        return_magnitude: If ``True``, return magnitude of the complex result.
        return_phase: If ``True``, return phase of the complex result.
        radial_bins: Number of radial bins for frequency aggregation.
            If ``None``, returns full 3D spectrum.
        max_frequency: Maximum normalized frequency to include (0.0 to 1.0].
        spatial_dims: Spatial dimensions to apply transform to.
            Default is last three dimensions.

    Returns:
        Radial Fourier transform of input data. Shape depends on parameters:
        - If ``radial_bins`` is ``None`` and only magnitude OR phase is requested:
          same spatial shape as input (..., D, H, W)
        - If ``radial_bins`` is ``None`` and both magnitude AND phase are requested:
          shape (..., D, H, 2*W) [magnitude and phase concatenated along last dimension]
        - If ``radial_bins`` is set and only magnitude OR phase is requested:
          shape (..., radial_bins)
        - If ``radial_bins`` is set and both magnitude AND phase are requested:
          shape (..., 2*radial_bins)

    Raises:
        ValueError: If ``max_frequency`` not in (0.0, 1.0], ``radial_bins`` < 1,
            or both ``return_magnitude`` and ``return_phase`` are ``False``.
    """

    def __init__(
        self,
        normalize: bool = True,
        return_magnitude: bool = True,
        return_phase: bool = False,
        radial_bins: Optional[int] = None,
        max_frequency: float = 1.0,
        spatial_dims: Union[int, Sequence[int]] = (-3, -2, -1),
    ) -> None:
        super().__init__()
        self.normalize = normalize
        self.return_magnitude = return_magnitude
        self.return_phase = return_phase
        self.radial_bins = radial_bins
        self.max_frequency = max_frequency
        if isinstance(spatial_dims, int):
            spatial_dims = (spatial_dims,)
        self.spatial_dims = tuple(spatial_dims)

        if not 0.0 < max_frequency <= 1.0:
            raise ValueError("max_frequency must be in (0.0, 1.0]")
        if radial_bins is not None and radial_bins < 1:
            raise ValueError("radial_bins must be >= 1")
        if not return_magnitude and not return_phase:
            raise ValueError("At least one of return_magnitude or return_phase must be True")

    def _compute_radial_coordinates(
        self, shape: tuple[int, ...], device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Compute normalized radial frequency coordinates.

        Args:
            shape: Spatial shape of the input (D, H, W).
            device: Device for the output tensor (defaults to CPU if None).

        Returns:
            Tensor of shape matching input spatial dims, containing radial distances
            from DC (zero-frequency) component (range ~0 to 0.5).
        """
        coords = []
        for dim_size in shape:
            freq = torch.fft.fftfreq(dim_size, device=device)
            coords.append(freq)
        mesh = torch.meshgrid(coords, indexing="ij")
        radial = torch.sqrt(sum(c**2 for c in mesh))
        return radial

    def _compute_radial_spectrum(self, spectrum: torch.Tensor, radial_coords: torch.Tensor) -> torch.Tensor:
        """
        Aggregate complex spectrum into radial bins.

        Args:
            spectrum: Flattened complex FFT spectrum.
            radial_coords: Flattened radial distances corresponding to spectrum.

        Returns:
            Complex tensor of shape (radial_bins,) with mean values per bin
            (or original spectrum if radial_bins is None).
        """
        if self.radial_bins is None:
            return spectrum

        max_r = self.max_frequency * 0.5
        bin_edges = torch.linspace(0, max_r, self.radial_bins + 1, device=spectrum.device)

        result_real = torch.zeros(self.radial_bins, dtype=spectrum.real.dtype, device=spectrum.device)
        result_imag = torch.zeros(self.radial_bins, dtype=spectrum.imag.dtype, device=spectrum.device)

        bin_indices = torch.bucketize(radial_coords, bin_edges[1:-1], right=False)

        for i in range(self.radial_bins):
            mask = bin_indices == i
            if mask.any():
                result_real[i] = spectrum.real[mask].mean()
                result_imag[i] = spectrum.imag[mask].mean()

        return torch.complex(result_real, result_imag)

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Apply 3D Radial Fourier Transform to input data.

        Args:
            img: Input medical image data. Expected shape: (..., D, H, W)
                 where D, H, W are spatial dimensions.

        Returns:
            Transformed data in radial frequency domain (ndarray or Tensor matching input type).

        Raises:
            ValueError: If input does not have exactly 3 spatial dimensions.
        """
        img_tensor, *_ = convert_data_type(img, torch.Tensor)
        spatial_shape = tuple(img_tensor.shape[d] for d in self.spatial_dims)
        if len(spatial_shape) != 3:
            raise ValueError("Expected 3 spatial dimensions")

        spectrum = fftn(ifftshift(img_tensor, dim=self.spatial_dims), dim=self.spatial_dims)
        spectrum = fftshift(spectrum, dim=self.spatial_dims)

        if self.normalize:
            norm_factor = math.prod(spatial_shape)
            spectrum = spectrum / norm_factor

        radial_coords = self._compute_radial_coordinates(spatial_shape, device=spectrum.device)

        if self.radial_bins is not None:
            orig_shape = spectrum.shape
            spatial_indices = [d % len(orig_shape) for d in self.spatial_dims]
            non_spatial_indices = [i for i in range(len(orig_shape)) if i not in spatial_indices]

            flat_shape = (*[orig_shape[i] for i in non_spatial_indices], -1)
            spectrum_flat = spectrum.moveaxis(spatial_indices, [-3, -2, -1]).reshape(flat_shape)
            radial_flat = radial_coords.flatten()

            non_spatial_dims = spectrum_flat.shape[:-1]
            non_spatial_product = math.prod(non_spatial_dims)
            spectrum_2d = spectrum_flat.reshape(non_spatial_product, -1)

            results = []
            for i in range(non_spatial_product):
                elem_spectrum = spectrum_2d[i]
                radial_result = self._compute_radial_spectrum(elem_spectrum, radial_flat)
                results.append(radial_result)

            spectrum = torch.stack(results, dim=0)
            spectrum = spectrum.reshape(*non_spatial_dims, self.radial_bins)
        else:
            if self.max_frequency < 1.0:
                freq_mask = radial_coords <= (self.max_frequency * 0.5)
                n_non_spatial = len(spectrum.shape) - len(spatial_shape)
                for _ in range(n_non_spatial):
                    freq_mask = freq_mask.unsqueeze(0)
                spectrum = spectrum * freq_mask

        output: Optional[torch.Tensor] = None
        if self.return_magnitude:
            magnitude = torch.abs(spectrum)
            output = magnitude if output is None else torch.cat([output, magnitude], dim=-1)
        if self.return_phase:
            phase = torch.angle(spectrum)
            output = phase if output is None else torch.cat([output, phase], dim=-1)

        output = cast(torch.Tensor, output)
        output, *_ = convert_data_type(output, type(img))
        return output

    def inverse(self, radial_data: NdarrayOrTensor, original_shape: tuple[int, ...]) -> NdarrayOrTensor:
        """
        Inverse transform from radial frequency domain to spatial domain.

        Args:
            radial_data: Data in radial frequency domain.
            original_shape: Original spatial shape (D, H, W).

        Returns:
            Reconstructed spatial data.

        Raises:
            ValueError: If input dimensions don't match expected shape for magnitude+phase.
            NotImplementedError: If radial_bins is not None.
        """
        if self.radial_bins is not None:
            raise NotImplementedError("Exact inverse not available for radially binned data.")

        radial_tensor, *_ = convert_data_type(radial_data, torch.Tensor)

        if self.return_magnitude and self.return_phase:
            last_dim = radial_tensor.shape[-1]
            if last_dim != original_shape[-1] * 2:
                raise ValueError("Expected last dimension to be doubled for magnitude+phase.")
            split_size = original_shape[-1]
            magnitude = radial_tensor[..., :split_size]
            phase = radial_tensor[..., split_size:]
            radial_tensor = torch.complex(magnitude * torch.cos(phase), magnitude * torch.sin(phase))

        result = ifftn(ifftshift(radial_tensor, dim=self.spatial_dims), dim=self.spatial_dims)
        result = fftshift(result, dim=self.spatial_dims)

        if self.normalize:
            result = result * math.prod(original_shape)

        result, *_ = convert_data_type(result.real, type(radial_data))
        return result


class RadialFourierFeatures3D(Transform):
    """
    Extract multi-scale radial Fourier features from 3D medical images.

    This transform composes multiple :class:`RadialFourier3D` instances with different
    radial bin counts to produce a concatenated feature vector. Useful for creating
    rotation-invariant frequency descriptors for downstream tasks like classification
    or registration.

    Args:
        n_bins_list: Sequence of radial bin counts to compute (e.g., (32, 64, 128)).
        return_types: Sequence of output types to compute per bin:
            ``"magnitude"``, ``"phase"``, or ``"complex"`` (both concatenated as real values).
        normalize: If ``True``, normalize FFT by the number of voxels.

    Returns:
        Concatenated 1D feature vector along the last dimension.
        Total feature size = sum over bins of (n_bins * factors based on return_types).

    Raises:
        ValueError: If ``n_bins_list`` or ``return_types`` is empty.
    """

    def __init__(
        self,
        n_bins_list: Sequence[int] = (32, 64, 128),
        return_types: Sequence[str] = ("magnitude",),
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.n_bins_list = n_bins_list
        self.return_types = return_types
        self.normalize = normalize

        if not n_bins_list:
            raise ValueError("n_bins_list must not be empty")
        if not return_types:
            raise ValueError("return_types must not be empty")

        self.transforms = []
        for n_bins in n_bins_list:
            for return_type in return_types:
                transform = RadialFourier3D(
                    normalize=normalize,
                    return_magnitude=(return_type in {"magnitude", "complex"}),
                    return_phase=(return_type in {"phase", "complex"}),
                    radial_bins=n_bins,
                )
                self.transforms.append(transform)

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Apply the composed radial Fourier transforms.

        Args:
            img: Input data with at least 3 spatial dimensions (..., D, H, W).

        Returns:
            Concatenated feature tensor matching input type (ndarray or Tensor).
        """
        features = [transform(img) for transform in self.transforms]
        features_tensors = [convert_data_type(feat, torch.Tensor)[0] for feat in features]
        output = torch.cat(features_tensors, dim=-1)
        output, *_ = convert_data_type(output, type(img))
        return output

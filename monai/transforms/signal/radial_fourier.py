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
"""
3D Radial Fourier Transform for medical imaging data.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional, Union

import numpy as np
import torch
from torch.fft import fftn, fftshift, ifftn, ifftshift

from monai.config import NdarrayOrTensor
from monai.transforms.transform import Transform
from monai.utils import convert_data_type

# Optional imports for type checking
# spatial, _ = optional_import("monai.utils", name="spatial")  # Commented out unused import


class RadialFourier3D(Transform):
    """
    Computes the 3D Radial Fourier Transform of medical imaging data.
    This transform converts 3D medical images into radial frequency domain representations,
    which is particularly useful for handling anisotropic resolution common in medical scans
    (e.g., different resolution in axial vs coronal planes).
    The radial transform provides rotation-invariant frequency analysis and can help
    normalize frequency representations across datasets with different acquisition parameters.
    Args:
        normalize: if True, normalize the output by the number of voxels.
        return_magnitude: if True, return magnitude of the complex result.
        return_phase: if True, return phase of the complex result.
        radial_bins: number of radial bins for frequency aggregation. If None, returns full 3D spectrum.
        max_frequency: maximum normalized frequency to include (0.0 to 1.0).
        spatial_dims: spatial dimensions to apply transform to. Default is last three dimensions.
    Returns:
        Radial Fourier transform of input data. Shape depends on parameters:
        - If radial_bins is None: complex tensor of same spatial shape as input
        - If radial_bins is set: real tensor of shape (radial_bins,) for magnitude/phase
    Example:
        >>> transform = RadialFourier3D(radial_bins=64, return_magnitude=True)
        >>> image = torch.randn(1, 128, 128, 96)  # Batch, Height, Width, Depth
        >>> result = transform(image)  # Shape: (1, 64)
    Raises:
        ValueError: If max_frequency not in (0.0, 1.0], radial_bins < 1, or both
            return_magnitude and return_phase are False.
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

        # Validate parameters
        if not 0.0 < max_frequency <= 1.0:
            raise ValueError(f"max_frequency must be in (0.0, 1.0], got {max_frequency}")
        if radial_bins is not None and radial_bins < 1:
            raise ValueError(f"radial_bins must be >= 1, got {radial_bins}")
        if not return_magnitude and not return_phase:
            raise ValueError("At least one of return_magnitude or return_phase must be True")

    def _compute_radial_coordinates(self, shape: tuple[int, ...], device: torch.device = None) -> torch.Tensor:
        """
        Compute radial distance from frequency domain center.

        Args:
            shape: spatial dimensions (D, H, W) or (H, W, D) depending on dims order.
            device: device to create tensor on.

        Returns:
            Tensor of same spatial shape with radial distances.
        """
        # Create frequency coordinates for each dimension
        coords = []
        for dim_size in shape:
            # Create frequency range from -0.5 to 0.5
            # Compatible with older PyTorch versions
            if hasattr(torch.fft, 'fftfreq'):
                freq = torch.fft.fftfreq(dim_size, device=device)
            else:
                # Fallback for older PyTorch versions (pre-1.8)
                n = dim_size
                val = 1.0 / n
                freq = torch.arange(-(n//2), (n+1)//2, device=device) * val
                freq = torch.roll(freq, n//2)
            coords.append(freq)

        # Create meshgrid and compute radial distance
        # Compatible with older PyTorch versions (pre-1.10)
        try:
            mesh = torch.meshgrid(coords, indexing="ij")
        except TypeError:
            # Older PyTorch doesn't support indexing parameter
            mesh = torch.meshgrid(coords)
            # Note: older meshgrid uses ij indexing by default in PyTorch

        radial = torch.sqrt(sum(c**2 for c in mesh))

        return radial

    def _compute_radial_spectrum(self, spectrum: torch.Tensor, radial_coords: torch.Tensor) -> torch.Tensor:
        """
        Compute radial average of frequency spectrum.

        Args:
            spectrum: complex frequency spectrum (flattened 1D array).
            radial_coords: radial distance for each frequency coordinate (flattened 1D array).

        Returns:
            Radial average of spectrum (1D array of length radial_bins).
        """
        if self.radial_bins is None:
            return spectrum

        # Bin radial coordinates
        max_r = self.max_frequency * 0.5  # Maximum normalized frequency
        bin_edges = torch.linspace(0, max_r, self.radial_bins + 1, device=spectrum.device)

        # Initialize output
        result_real = torch.zeros(self.radial_bins, dtype=spectrum.real.dtype, device=spectrum.device)
        result_imag = torch.zeros(self.radial_bins, dtype=spectrum.imag.dtype, device=spectrum.device)

        # Bin the frequencies - spectrum and radial_coords are both 1D
        for i in range(self.radial_bins):
            mask = (radial_coords >= bin_edges[i]) & (radial_coords < bin_edges[i + 1])
            if mask.any():
                # spectrum is 1D, so we can index it directly
                result_real[i] = spectrum.real[mask].mean()
                result_imag[i] = spectrum.imag[mask].mean()

        # Combine real and imaginary parts
        result = torch.complex(result_real, result_imag)

        return result

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Apply 3D Radial Fourier Transform to input data.

        Args:
            img: input medical image data. Expected shape: (..., D, H, W)
                 where D, H, W are spatial dimensions.

        Returns:
            Transformed data in radial frequency domain.
        """
        # Convert to tensor if needed
        img_tensor, *_ = convert_data_type(img, torch.Tensor)
        # Get spatial dimensions
        spatial_shape = tuple(img_tensor.shape[d] for d in self.spatial_dims)
        if len(spatial_shape) != 3:
            raise ValueError(f"Expected 3 spatial dimensions, got {len(spatial_shape)}")

        # Compute 3D FFT
        # Shift zero frequency to center and compute FFT
        spectrum = fftn(ifftshift(img_tensor, dim=self.spatial_dims), dim=self.spatial_dims)
        spectrum = fftshift(spectrum, dim=self.spatial_dims)

        # Normalize if requested
        if self.normalize:
            norm_factor = 1
            for dim in spatial_shape:
                norm_factor *= dim
            spectrum = spectrum / norm_factor

        # Compute radial coordinates
        radial_coords = self._compute_radial_coordinates(spatial_shape, device=spectrum.device)

        # Apply radial binning if requested
        if self.radial_bins is not None:
            # Reshape for radial processing
            orig_shape = spectrum.shape
            # Move spatial dimensions to end for processing
            spatial_indices = [d % len(orig_shape) for d in self.spatial_dims]
            non_spatial_indices = [i for i in range(len(orig_shape)) if i not in spatial_indices]

            # Reshape to (non_spatial..., spatial_prod)
            flat_shape = (*[orig_shape[i] for i in non_spatial_indices], -1)
            spectrum_flat = spectrum.moveaxis(spatial_indices, [-3, -2, -1]).reshape(flat_shape)
            radial_flat = radial_coords.flatten()

            # Get non-spatial dimensions (batch, channel, etc.)
            non_spatial_dims = spectrum_flat.shape[:-1]
            spatial_size = spectrum_flat.shape[-1]

            # Reshape to 2D: (non_spatial_product, spatial_size)
            non_spatial_product = 1
            for dim in non_spatial_dims:
                non_spatial_product *= dim

            spectrum_2d = spectrum_flat.reshape(non_spatial_product, spatial_size)

            # Process each non-spatial element (batch/channel combination)
            results = []
            for i in range(non_spatial_product):
                elem_spectrum = spectrum_2d[i]  # Get spatial frequencies for this batch/channel
                radial_result = self._compute_radial_spectrum(elem_spectrum, radial_flat)
                results.append(radial_result)

            # Combine results and reshape back
            spectrum = torch.stack(results, dim=0)
            spectrum = spectrum.reshape(*non_spatial_dims, self.radial_bins)
        else:
            # Apply frequency mask if max_frequency < 1.0
            if self.max_frequency < 1.0:
                freq_mask = radial_coords <= (self.max_frequency * 0.5)
                # Expand mask to match spectrum dimensions
                n_non_spatial = len(spectrum.shape) - len(spatial_shape)
                for _ in range(n_non_spatial):
                    freq_mask = freq_mask.unsqueeze(0)
                spectrum = spectrum * freq_mask

        # Extract magnitude and/or phase as requested
        output = None
        if self.return_magnitude:
            magnitude = torch.abs(spectrum)
            output = magnitude if output is None else torch.cat([output, magnitude], dim=-1)

        if self.return_phase:
            phase = torch.angle(spectrum)
            output = phase if output is None else torch.cat([output, phase], dim=-1)

        # Convert back to original data type
        output, *_ = convert_data_type(output, type(img))

        return output

    def inverse(self, radial_data: NdarrayOrTensor, original_shape: tuple[int, ...]) -> NdarrayOrTensor:
        """
        Inverse transform from radial frequency domain to spatial domain.

        Args:
            radial_data: data in radial frequency domain.
            original_shape: original spatial shape (D, H, W).

        Returns:
            Reconstructed spatial data.

        Note:
            This is an approximate inverse when radial_bins is used.
        """
        if self.radial_bins is None:
            # Direct inverse FFT
            radial_tensor, *_ = convert_data_type(radial_data, torch.Tensor)

            # Separate magnitude and phase if needed
            if self.return_magnitude and self.return_phase:
                # Assuming they were concatenated along last dimension
                split_idx = radial_tensor.shape[-1] // 2
                magnitude = radial_tensor[..., :split_idx]
                phase = radial_tensor[..., split_idx:]
                radial_tensor = torch.complex(magnitude * torch.cos(phase), magnitude * torch.sin(phase))

            # Apply inverse FFT
            result = ifftn(ifftshift(radial_tensor, dim=self.spatial_dims), dim=self.spatial_dims)
            result = fftshift(result, dim=self.spatial_dims)

            if self.normalize:
                shape_product = 1
                for dim in original_shape:
                    shape_product *= dim
                result = result * shape_product

            result, *_ = convert_data_type(result.real, type(radial_data))
            return result

        else:
            raise NotImplementedError(
                "Exact inverse transform not available for radially binned data. "
                "Consider using radial_bins=None for applications requiring inversion."
            )


class RadialFourierFeatures3D(Transform):
    """
    Extract radial Fourier features for medical image analysis.
    Computes multiple radial Fourier transforms with different parameters
    to create a comprehensive frequency feature representation.
    Args:
        n_bins_list: list of radial bin counts to compute.
        return_types: list of return types: 'magnitude', 'phase', or 'complex'.
        normalize: if True, normalize the output.
    Returns:
        Concatenated radial Fourier features.
    Example:
        >>> transform = RadialFourierFeatures3D(n_bins_list=[32, 64, 128])
        >>> image = torch.randn(1, 128, 128, 96)
        >>> features = transform(image)  # Shape: (1, 32+64+128=224)
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

        # Create individual transforms
        self.transforms = []
        for n_bins in n_bins_list:
            for return_type in return_types:
                transform = RadialFourier3D(
                    normalize=normalize,
                    return_magnitude=(return_type in ["magnitude", "complex"]),
                    return_phase=(return_type in ["phase", "complex"]),
                    radial_bins=n_bins,
                )
                self.transforms.append(transform)

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        """Extract radial Fourier features."""
        features = []
        for transform in self.transforms:
            feat = transform(img)
            features.append(feat)

        # Concatenate along last dimension
        if features:
            # Convert all features to tensors if any are numpy arrays
            features_tensors = []
            for feat in features:
                if isinstance(feat, np.ndarray):
                    features_tensors.append(torch.from_numpy(feat))
                else:
                    features_tensors.append(feat)
            output = torch.cat(features_tensors, dim=-1)
        else:
            output = img

        # Convert to original type if needed
        if isinstance(img, np.ndarray):
            output = output.cpu().numpy()

        return output

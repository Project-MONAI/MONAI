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

from typing import TYPE_CHECKING

import numpy as np
import torch

from monai.data import ITKReader
from monai.data.meta_tensor import MetaTensor
from monai.transforms import EnsureChannelFirst
from monai.utils import convert_to_dst_type, optional_import

if TYPE_CHECKING:
    import itk

    has_itk = True
else:
    itk, has_itk = optional_import("itk")

__all__ = [
    "itk_image_to_metatensor",
    "itk_to_monai_affine",
    "monai_to_itk_affine",
    "monai_to_itk_ddf"
]


def itk_image_to_metatensor(image):
    """
    Converts an ITK image to a MetaTensor object.

    Args:
        image: The ITK image to be converted.

    Returns:
        A MetaTensor object containing the array data and metadata.
    """
    reader = ITKReader(affine_lps_to_ras=False)
    image_array, meta_data = reader.get_data(image)
    image_array = convert_to_dst_type(image_array, dst=image_array, dtype=np.dtype(itk.D))[0]
    metatensor = MetaTensor.ensure_torch_and_prune_meta(image_array, meta_data)
    metatensor = EnsureChannelFirst()(metatensor)

    return metatensor


def _compute_offset_matrix(image, center_of_rotation) -> tuple[torch.Tensor, torch.Tensor]:
    ndim = image.ndim
    offset = np.asarray(_get_itk_image_center(image)) - np.asarray(center_of_rotation)
    offset_matrix = torch.eye(ndim + 1, dtype=torch.float64)
    offset_matrix[:ndim, ndim] = torch.tensor(offset, dtype=torch.float64)
    inverse_offset_matrix = torch.eye(ndim + 1, dtype=torch.float64)
    inverse_offset_matrix[:ndim, ndim] = -torch.tensor(offset, dtype=torch.float64)

    return offset_matrix, inverse_offset_matrix


def _compute_spacing_matrix(image):
    ndim = image.ndim
    spacing = np.asarray(image.GetSpacing(), dtype=np.float64)
    spacing_matrix = torch.eye(ndim + 1, dtype=torch.float64)
    inverse_spacing_matrix = torch.eye(ndim + 1, dtype=torch.float64)
    for i, e in enumerate(spacing):
        spacing_matrix[i, i] = e
        inverse_spacing_matrix[i, i] = 1 / e

    return spacing_matrix, inverse_spacing_matrix


def _compute_direction_matrix(image):
    ndim = image.ndim
    direction = itk.array_from_matrix(image.GetDirection())
    direction_matrix = torch.eye(ndim + 1, dtype=torch.float64)
    direction_matrix[:ndim, :ndim] = torch.tensor(direction, dtype=torch.float64)
    inverse_direction = itk.array_from_matrix(image.GetInverseDirection())
    inverse_direction_matrix = torch.eye(ndim + 1, dtype=torch.float64)
    inverse_direction_matrix[:ndim, :ndim] = torch.tensor(inverse_direction, dtype=torch.float64)

    return direction_matrix, inverse_direction_matrix


def itk_to_monai_affine(image, matrix, translation, center_of_rotation=None) -> torch.Tensor:
    """
    Converts an ITK affine matrix (2x2 for 2D or 3x3 for 3D matrix and translation vector) to a MONAI affine matrix.

    Args:
        image: The ITK image object. This is used to extract the spacing and direction information.
        matrix: The 2x2 or 3x3 ITK affine matrix.
        translation: The 2-element or 3-element ITK affine translation vector.
        center_of_rotation: The center of rotation. If provided, the affine
                            matrix will be adjusted to account for the difference
                            between the center of the image and the center of rotation.

    Returns:
        A 4x4 MONAI affine matrix.
    """

    # Create affine matrix that includes translation
    ndim = image.ndim
    affine_matrix = torch.eye(ndim + 1, dtype=torch.float64)
    affine_matrix[:ndim, :ndim] = torch.tensor(matrix, dtype=torch.float64)
    affine_matrix[:ndim, ndim] = torch.tensor(translation, dtype=torch.float64)

    # Adjust offset when center of rotation is different from center of the image
    if center_of_rotation:
        offset_matrix, inverse_offset_matrix = _compute_offset_matrix(image, center_of_rotation)
        affine_matrix = inverse_offset_matrix @ affine_matrix @ offset_matrix

    # Adjust based on spacing. It is required because MONAI does not update the
    # pixel array according to the spacing after a transformation. For example,
    # a rotation of 90deg for an image with different spacing along the two axis
    # will just rotate the image array by 90deg without also scaling accordingly.
    # TODO rewrite comment
    spacing_matrix, inverse_spacing_matrix = _compute_spacing_matrix(image)
    affine_matrix = inverse_spacing_matrix @ affine_matrix @ spacing_matrix

    # Adjust direction
    direction_matrix, inverse_direction_matrix = _compute_direction_matrix(image)
    affine_matrix = inverse_direction_matrix @ affine_matrix @ direction_matrix

    return affine_matrix


def monai_to_itk_affine(image, affine_matrix, center_of_rotation=None):
    """
    Converts a MONAI affine matrix to an ITK affine matrix (2x2 for 2D or 3x3 for
    3D matrix and translation vector). See also 'itk_to_monai_affine'.

    Args:
        image: The ITK image object. This is used to extract the spacing and direction information.
        affine_matrix: The 3x3 for 2D or 4x4 for 3D MONAI affine matrix.
        center_of_rotation: The center of rotation. If provided, the affine
                            matrix will be adjusted to account for the difference
                            between the center of the image and the center of rotation.

    Returns:
        The ITK matrix and the translation vector.
    """
    # Adjust direction
    direction_matrix, inverse_direction_matrix = _compute_direction_matrix(image)
    affine_matrix = direction_matrix @ affine_matrix @ inverse_direction_matrix

    # Adjust spacing
    spacing_matrix, inverse_spacing_matrix = _compute_spacing_matrix(image)
    affine_matrix = spacing_matrix @ affine_matrix @ inverse_spacing_matrix

    # Adjust offset when center of rotation is different from center of the image
    if center_of_rotation:
        offset_matrix, inverse_offset_matrix = _compute_offset_matrix(image, center_of_rotation)
        affine_matrix = offset_matrix @ affine_matrix @ inverse_offset_matrix

    ndim = image.ndim
    matrix = affine_matrix[:ndim, :ndim].numpy()
    translation = affine_matrix[:ndim, ndim].tolist()

    return matrix, translation


def _get_itk_image_center(image):
    """
    Calculates the center of the ITK image based on its origin, size, and spacing.
    This center is equivalent to the implicit image center that MONAI uses.

    Args:
        image: The ITK image.

    Returns:
        The center of the image as a list of coordinates.
    """
    image_size = np.asarray(image.GetLargestPossibleRegion().GetSize(), np.float32)
    spacing = np.asarray(image.GetSpacing())
    origin = np.asarray(image.GetOrigin())
    center = image.GetDirection() @ ((image_size / 2 - 0.5) * spacing) + origin

    return center.tolist()


def monai_to_itk_ddf(image, ddf):
    """
    converting the dense displacement field from the MONAI space to the ITK
    Args:
        image: itk image of array shape 2D: (H, W) or 3D: (D, H, W)
        ddf: numpy array of shape 2D: (2, H, W) or 3D: (3, D, H, W)
    Returns:
        displacement_field: itk image of the corresponding displacement field

    """
    # 3, D, H, W -> D, H, W, 3
    ndim = image.ndim
    ddf = ddf.transpose(tuple(list(range(1, ndim + 1)) + [0]))
    # x, y, z -> z, x, y
    ddf = ddf[..., ::-1]

    # Correct for spacing
    spacing = np.asarray(image.GetSpacing(), dtype=np.float64)
    ddf *= np.array(spacing, ndmin=ndim + 1)

    # Correct for direction
    direction = np.asarray(image.GetDirection(), dtype=np.float64)
    ddf = np.einsum("ij,...j->...i", direction, ddf, dtype=np.float64).astype(np.float32)

    # initialise displacement field -
    vector_component_type = itk.F
    vector_pixel_type = itk.Vector[vector_component_type, ndim]
    displacement_field_type = itk.Image[vector_pixel_type, ndim]
    displacement_field = itk.GetImageFromArray(ddf, ttype=displacement_field_type)

    # Set image metadata
    displacement_field.SetSpacing(image.GetSpacing())
    displacement_field.SetOrigin(image.GetOrigin())
    displacement_field.SetDirection(image.GetDirection())

    return displacement_field

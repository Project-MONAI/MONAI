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

from typing import TYPE_CHECKING, cast

import numpy as np
import torch

from monai.config.type_definitions import DtypeLike
from monai.data import ITKReader, ITKWriter
from monai.data.meta_tensor import MetaTensor
from monai.data.utils import orientation_ras_lps
from monai.transforms import EnsureChannelFirst
from monai.utils import MetaKeys, SpaceKeys, convert_to_dst_type, optional_import

if TYPE_CHECKING:
    import itk

    has_itk = True
else:
    itk, has_itk = optional_import("itk")

__all__ = [
    "itk_image_to_metatensor",
    "metatensor_to_itk_image",
    "itk_to_monai_affine",
    "monai_to_itk_affine",
    "get_itk_image_center",
    "monai_to_itk_ddf",
]


def itk_image_to_metatensor(
    image, channel_dim: str | int | None = None, dtype: DtypeLike | torch.dtype = float
) -> MetaTensor:
    """
    Converts an ITK image to a MetaTensor object.

    Args:
        image: The ITK image to be converted.
        channel_dim: the channel dimension of the input image, default is None.
            This is used to set original_channel_dim in the metadata, EnsureChannelFirst reads this field.
            If None, the channel_dim is inferred automatically.
            If the input array doesn't have a channel dim, this value should be ``'no_channel'``.
        dtype: output dtype, defaults to the Python built-in `float`.

    Returns:
        A MetaTensor object containing the array data and metadata in ChannelFirst format.
    """
    reader = ITKReader(affine_lps_to_ras=False, channel_dim=channel_dim)
    image_array, meta_data = reader.get_data(image)
    image_array = convert_to_dst_type(image_array, dst=image_array, dtype=dtype)[0]
    metatensor = MetaTensor.ensure_torch_and_prune_meta(image_array, meta_data)
    metatensor = EnsureChannelFirst(channel_dim=channel_dim)(metatensor)

    return cast(MetaTensor, metatensor)


def metatensor_to_itk_image(
    meta_tensor: MetaTensor, channel_dim: int | None = 0, dtype: DtypeLike = np.float32, **kwargs
):
    """
    Converts a MetaTensor object to an ITK image. Expects the MetaTensor to be in ChannelFirst format.

    Args:
        meta_tensor: The MetaTensor to be converted.
        channel_dim: channel dimension of the data array, defaults to ``0`` (Channel-first).
            ``None`` indicates no channel dimension. This is used to create a Vector Image if it is not ``None``.
        dtype: output data type, defaults to `np.float32`.
        kwargs: additional keyword arguments. Currently `itk.GetImageFromArray` will get ``ttype`` from this dictionary.

    Returns:
        The ITK image.

    See also: :py:func:`ITKWriter.create_backend_obj`
    """
    if meta_tensor.meta.get(MetaKeys.SPACE, SpaceKeys.LPS) == SpaceKeys.RAS:
        _meta_tensor = meta_tensor.clone()
        _meta_tensor.affine = orientation_ras_lps(meta_tensor.affine)
        _meta_tensor.meta[MetaKeys.SPACE] = SpaceKeys.LPS
    else:
        _meta_tensor = meta_tensor
    writer = ITKWriter(output_dtype=dtype, affine_lps_to_ras=False)
    writer.set_data_array(data_array=meta_tensor.data, channel_dim=channel_dim, squeeze_end_dims=True)
    return writer.create_backend_obj(
        writer.data_obj,
        channel_dim=writer.channel_dim,
        affine=_meta_tensor.affine,
        affine_lps_to_ras=False,  # False if the affine is in itk convention
        dtype=writer.output_dtype,
        kwargs=kwargs,
    )


def itk_to_monai_affine(image, matrix, translation, center_of_rotation=None, reference_image=None) -> torch.Tensor:
    """
    Converts an ITK affine matrix (2x2 for 2D or 3x3 for 3D matrix and translation vector) to a MONAI affine matrix.

    Args:
        image: The ITK image object. This is used to extract the spacing and direction information.
        matrix: The 2x2 or 3x3 ITK affine matrix.
        translation: The 2-element or 3-element ITK affine translation vector.
        center_of_rotation: The center of rotation. If provided, the affine
                            matrix will be adjusted to account for the difference
                            between the center of the image and the center of rotation.
        reference_image: The coordinate space that matrix and translation were defined
                         in respect to. If not supplied, the coordinate space of image
                         is used.

    Returns:
        A 4x4 MONAI affine matrix.
    """

    _assert_itk_regions_match_array(image)
    ndim = image.ndim
    # If there is a reference image, compute an affine matrix that maps the image space to the
    # reference image space.
    if reference_image:
        reference_affine_matrix = _compute_reference_space_affine_matrix(image, reference_image)
    else:
        reference_affine_matrix = torch.eye(ndim + 1, dtype=torch.float64)

    # Create affine matrix that includes translation
    affine_matrix = torch.eye(ndim + 1, dtype=torch.float64)
    affine_matrix[:ndim, :ndim] = torch.tensor(matrix, dtype=torch.float64)
    affine_matrix[:ndim, ndim] = torch.tensor(translation, dtype=torch.float64)

    # Adjust offset when center of rotation is different from center of the image
    if center_of_rotation:
        offset_matrix, inverse_offset_matrix = _compute_offset_matrix(image, center_of_rotation)
        affine_matrix = inverse_offset_matrix @ affine_matrix @ offset_matrix

    # Adjust direction
    direction_matrix, inverse_direction_matrix = _compute_direction_matrix(image)
    affine_matrix = inverse_direction_matrix @ affine_matrix @ direction_matrix

    # Adjust based on spacing. It is required because MONAI does not update the
    # pixel array according to the spacing after a transformation. For example,
    # a rotation of 90deg for an image with different spacing along the two axis
    # will just rotate the image array by 90deg without also scaling accordingly.
    spacing_matrix, inverse_spacing_matrix = _compute_spacing_matrix(image)
    affine_matrix = inverse_spacing_matrix @ affine_matrix @ spacing_matrix

    return affine_matrix @ reference_affine_matrix


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
    _assert_itk_regions_match_array(image)

    # Adjust spacing
    spacing_matrix, inverse_spacing_matrix = _compute_spacing_matrix(image)
    affine_matrix = spacing_matrix @ affine_matrix @ inverse_spacing_matrix

    # Adjust direction
    direction_matrix, inverse_direction_matrix = _compute_direction_matrix(image)
    affine_matrix = direction_matrix @ affine_matrix @ inverse_direction_matrix

    # Adjust offset when center of rotation is different from center of the image
    if center_of_rotation:
        offset_matrix, inverse_offset_matrix = _compute_offset_matrix(image, center_of_rotation)
        affine_matrix = offset_matrix @ affine_matrix @ inverse_offset_matrix

    ndim = image.ndim
    matrix = affine_matrix[:ndim, :ndim].numpy()
    translation = affine_matrix[:ndim, ndim].tolist()

    return matrix, translation


def get_itk_image_center(image):
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


def _assert_itk_regions_match_array(image):
    # Note: Make it more compact? Also, are there redundant checks?
    largest_region = image.GetLargestPossibleRegion()
    buffered_region = image.GetBufferedRegion()
    requested_region = image.GetRequestedRegion()

    largest_region_size = np.array(largest_region.GetSize())
    buffered_region_size = np.array(buffered_region.GetSize())
    requested_region_size = np.array(requested_region.GetSize())
    array_size = np.array(image.shape)[::-1]

    largest_region_index = np.array(largest_region.GetIndex())
    buffered_region_index = np.array(buffered_region.GetIndex())
    requested_region_index = np.array(requested_region.GetIndex())

    indices_are_zeros = (
        np.all(largest_region_index == 0) and np.all(buffered_region_index == 0) and np.all(requested_region_index == 0)
    )

    sizes_match = (
        np.array_equal(array_size, largest_region_size)
        and np.array_equal(largest_region_size, buffered_region_size)
        and np.array_equal(buffered_region_size, requested_region_size)
    )

    if not indices_are_zeros:
        raise AssertionError("ITK-MONAI bridge: non-zero ITK region indices encountered")
    if not sizes_match:
        raise AssertionError("ITK-MONAI bridge: ITK regions should be of the same shape")


def _compute_offset_matrix(image, center_of_rotation) -> tuple[torch.Tensor, torch.Tensor]:
    ndim = image.ndim
    offset = np.asarray(get_itk_image_center(image)) - np.asarray(center_of_rotation)
    offset_matrix = torch.eye(ndim + 1, dtype=torch.float64)
    offset_matrix[:ndim, ndim] = torch.tensor(offset, dtype=torch.float64)
    inverse_offset_matrix = torch.eye(ndim + 1, dtype=torch.float64)
    inverse_offset_matrix[:ndim, ndim] = -torch.tensor(offset, dtype=torch.float64)

    return offset_matrix, inverse_offset_matrix


def _compute_spacing_matrix(image) -> tuple[torch.Tensor, torch.Tensor]:
    ndim = image.ndim
    spacing = np.asarray(image.GetSpacing(), dtype=np.float64)
    spacing_matrix = torch.eye(ndim + 1, dtype=torch.float64)
    inverse_spacing_matrix = torch.eye(ndim + 1, dtype=torch.float64)
    for i, e in enumerate(spacing):
        spacing_matrix[i, i] = e
        inverse_spacing_matrix[i, i] = 1 / e

    return spacing_matrix, inverse_spacing_matrix


def _compute_direction_matrix(image) -> tuple[torch.Tensor, torch.Tensor]:
    ndim = image.ndim
    direction = itk.array_from_matrix(image.GetDirection())
    direction_matrix = torch.eye(ndim + 1, dtype=torch.float64)
    direction_matrix[:ndim, :ndim] = torch.tensor(direction, dtype=torch.float64)
    inverse_direction = itk.array_from_matrix(image.GetInverseDirection())
    inverse_direction_matrix = torch.eye(ndim + 1, dtype=torch.float64)
    inverse_direction_matrix[:ndim, :ndim] = torch.tensor(inverse_direction, dtype=torch.float64)

    return direction_matrix, inverse_direction_matrix


def _compute_reference_space_affine_matrix(image, ref_image) -> torch.Tensor:
    ndim = ref_image.ndim

    # Spacing and direction as matrices
    spacing_matrix, inv_spacing_matrix = (m[:ndim, :ndim].numpy() for m in _compute_spacing_matrix(image))
    ref_spacing_matrix, ref_inv_spacing_matrix = (m[:ndim, :ndim].numpy() for m in _compute_spacing_matrix(ref_image))

    direction_matrix, inv_direction_matrix = (m[:ndim, :ndim].numpy() for m in _compute_direction_matrix(image))
    ref_direction_matrix, ref_inv_direction_matrix = (
        m[:ndim, :ndim].numpy() for m in _compute_direction_matrix(ref_image)
    )

    # Matrix calculation
    matrix = ref_direction_matrix @ ref_spacing_matrix @ inv_spacing_matrix @ inv_direction_matrix

    # Offset calculation
    pixel_offset = -1
    image_size = np.asarray(ref_image.GetLargestPossibleRegion().GetSize(), np.float32)
    translation = (
        (ref_direction_matrix @ ref_spacing_matrix - direction_matrix @ spacing_matrix)
        @ (image_size + pixel_offset)
        / 2
    )
    translation += np.asarray(ref_image.GetOrigin()) - np.asarray(image.GetOrigin())

    # Convert matrix ITK matrix and translation to MONAI affine matrix
    ref_affine_matrix = itk_to_monai_affine(image, matrix=matrix, translation=translation)

    return ref_affine_matrix


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

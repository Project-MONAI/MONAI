# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import itertools
import random
import warnings
from contextlib import contextmanager
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from monai.config import DtypeLike, IndexSelection
from monai.networks.layers import GaussianFilter
from monai.transforms.compose import Compose
from monai.transforms.transform import MapTransform
from monai.utils import (
    GridSampleMode,
    InterpolateMode,
    InverseKeys,
    ensure_tuple,
    ensure_tuple_rep,
    ensure_tuple_size,
    fall_back_tuple,
    issequenceiterable,
    min_version,
    optional_import,
)

measure, _ = optional_import("skimage.measure", "0.14.2", min_version)

__all__ = [
    "rand_choice",
    "img_bounds",
    "in_bounds",
    "is_empty",
    "zero_margins",
    "rescale_array",
    "rescale_instance_array",
    "rescale_array_int_max",
    "copypaste_arrays",
    "resize_center",
    "map_binary_to_indices",
    "weighted_patch_samples",
    "generate_pos_neg_label_crop_centers",
    "create_grid",
    "create_control_grid",
    "create_rotate",
    "create_shear",
    "create_scale",
    "create_translate",
    "generate_spatial_bounding_box",
    "get_largest_connected_component_mask",
    "get_extreme_points",
    "extreme_points_to_image",
    "map_spatial_axes",
    "allow_missing_keys_mode",
    "convert_inverse_interp_mode",
]


def rand_choice(prob: float = 0.5) -> bool:
    """
    Returns True if a randomly chosen number is less than or equal to `prob`, by default this is a 50/50 chance.
    """
    return bool(random.random() <= prob)


def img_bounds(img: np.ndarray):
    """
    Returns the minimum and maximum indices of non-zero lines in axis 0 of `img`, followed by that for axis 1.
    """
    ax0 = np.any(img, axis=0)
    ax1 = np.any(img, axis=1)
    return np.concatenate((np.where(ax0)[0][[0, -1]], np.where(ax1)[0][[0, -1]]))


def in_bounds(x: float, y: float, margin: float, maxx: float, maxy: float) -> bool:
    """
    Returns True if (x,y) is within the rectangle (margin, margin, maxx-margin, maxy-margin).
    """
    return bool(margin <= x < (maxx - margin) and margin <= y < (maxy - margin))


def is_empty(img: Union[np.ndarray, torch.Tensor]) -> bool:
    """
    Returns True if `img` is empty, that is its maximum value is not greater than its minimum.
    """
    return not (img.max() > img.min())  # use > instead of <= so that an image full of NaNs will result in True


def zero_margins(img: np.ndarray, margin: int) -> bool:
    """
    Returns True if the values within `margin` indices of the edges of `img` in dimensions 1 and 2 are 0.
    """
    if np.any(img[:, :, :margin]) or np.any(img[:, :, -margin:]):
        return False

    return not np.any(img[:, :margin, :]) and not np.any(img[:, -margin:, :])


def rescale_array(arr: np.ndarray, minv: float = 0.0, maxv: float = 1.0, dtype: DtypeLike = np.float32):
    """
    Rescale the values of numpy array `arr` to be from `minv` to `maxv`.
    """
    if dtype is not None:
        arr = arr.astype(dtype)

    mina = np.min(arr)
    maxa = np.max(arr)

    if mina == maxa:
        return arr * minv

    norm = (arr - mina) / (maxa - mina)  # normalize the array first
    return (norm * (maxv - minv)) + minv  # rescale by minv and maxv, which is the normalized array by default


def rescale_instance_array(
    arr: np.ndarray, minv: float = 0.0, maxv: float = 1.0, dtype: DtypeLike = np.float32
) -> np.ndarray:
    """
    Rescale each array slice along the first dimension of `arr` independently.
    """
    out: np.ndarray = np.zeros(arr.shape, dtype)
    for i in range(arr.shape[0]):
        out[i] = rescale_array(arr[i], minv, maxv, dtype)

    return out


def rescale_array_int_max(arr: np.ndarray, dtype: DtypeLike = np.uint16) -> np.ndarray:
    """
    Rescale the array `arr` to be between the minimum and maximum values of the type `dtype`.
    """
    info: np.iinfo = np.iinfo(dtype)
    return np.asarray(rescale_array(arr, info.min, info.max), dtype=dtype)


def copypaste_arrays(
    src_shape,
    dest_shape,
    srccenter: Sequence[int],
    destcenter: Sequence[int],
    dims: Sequence[Optional[int]],
) -> Tuple[Tuple[slice, ...], Tuple[slice, ...]]:
    """
    Calculate the slices to copy a sliced area of array in `src_shape` into array in `dest_shape`.

    The area has dimensions `dims` (use 0 or None to copy everything in that dimension),
    the source area is centered at `srccenter` index in `src` and copied into area centered at `destcenter` in `dest`.
    The dimensions of the copied area will be clipped to fit within the
    source and destination arrays so a smaller area may be copied than expected. Return value is the tuples of slice
    objects indexing the copied area in `src`, and those indexing the copy area in `dest`.

    Example

    .. code-block:: python

        src_shape = (6,6)
        src = np.random.randint(0,10,src_shape)
        dest = np.zeros_like(src)
        srcslices, destslices = copypaste_arrays(src_shape, dest.shape, (3, 2),(2, 1),(3, 4))
        dest[destslices] = src[srcslices]
        print(src)
        print(dest)

        >>> [[9 5 6 6 9 6]
             [4 3 5 6 1 2]
             [0 7 3 2 4 1]
             [3 0 0 1 5 1]
             [9 4 7 1 8 2]
             [6 6 5 8 6 7]]
            [[0 0 0 0 0 0]
             [7 3 2 4 0 0]
             [0 0 1 5 0 0]
             [4 7 1 8 0 0]
             [0 0 0 0 0 0]
             [0 0 0 0 0 0]]

    """
    s_ndim = len(src_shape)
    d_ndim = len(dest_shape)
    srcslices = [slice(None)] * s_ndim
    destslices = [slice(None)] * d_ndim

    for i, ss, ds, sc, dc, dim in zip(range(s_ndim), src_shape, dest_shape, srccenter, destcenter, dims):
        if dim:
            # dimension before midpoint, clip to size fitting in both arrays
            d1 = np.clip(dim // 2, 0, min(sc, dc))
            # dimension after midpoint, clip to size fitting in both arrays
            d2 = np.clip(dim // 2 + 1, 0, min(ss - sc, ds - dc))

            srcslices[i] = slice(sc - d1, sc + d2)
            destslices[i] = slice(dc - d1, dc + d2)

    return tuple(srcslices), tuple(destslices)


def resize_center(img: np.ndarray, *resize_dims: Optional[int], fill_value: float = 0.0, inplace: bool = True):
    """
    Resize `img` by cropping or expanding the image from the center. The `resize_dims` values are the output dimensions
    (or None to use original dimension of `img`). If a dimension is smaller than that of `img` then the result will be
    cropped and if larger padded with zeros, in both cases this is done relative to the center of `img`. The result is
    a new image with the specified dimensions and values from `img` copied into its center.
    """

    resize_dims = fall_back_tuple(resize_dims, img.shape)

    half_img_shape = np.asarray(img.shape) // 2
    half_dest_shape = np.asarray(resize_dims) // 2
    srcslices, destslices = copypaste_arrays(img.shape, resize_dims, half_img_shape, half_dest_shape, resize_dims)

    if not inplace:
        dest = np.full(resize_dims, fill_value, img.dtype)  # type: ignore
        dest[destslices] = img[srcslices]
        return dest
    return img[srcslices]


def map_binary_to_indices(
    label: np.ndarray,
    image: Optional[np.ndarray] = None,
    image_threshold: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the foreground and background of input label data, return the indices after fattening.
    For example:
    ``label = np.array([[[0, 1, 1], [1, 0, 1], [1, 1, 0]]])``
    ``foreground indices = np.array([1, 2, 3, 5, 6, 7])`` and ``background indices = np.array([0, 4, 8])``

    Args:
        label: use the label data to get the foreground/background information.
        image: if image is not None, use ``label = 0 & image > image_threshold``
            to define background. so the output items will not map to all the voxels in the label.
        image_threshold: if enabled `image`, use ``image > image_threshold`` to
            determine the valid image content area and select background only in this area.

    """
    # Prepare fg/bg indices
    if label.shape[0] > 1:
        label = label[1:]  # for One-Hot format data, remove the background channel
    label_flat = np.any(label, axis=0).ravel()  # in case label has multiple dimensions
    fg_indices = np.nonzero(label_flat)[0]
    if image is not None:
        img_flat = np.any(image > image_threshold, axis=0).ravel()
        bg_indices = np.nonzero(np.logical_and(img_flat, ~label_flat))[0]
    else:
        bg_indices = np.nonzero(~label_flat)[0]
    return fg_indices, bg_indices


def weighted_patch_samples(
    spatial_size: Union[int, Sequence[int]],
    w: np.ndarray,
    n_samples: int = 1,
    r_state: Optional[np.random.RandomState] = None,
) -> List:
    """
    Computes `n_samples` of random patch sampling locations, given the sampling weight map `w` and patch `spatial_size`.

    Args:
        spatial_size: length of each spatial dimension of the patch.
        w: weight map, the weights must be non-negative. each element denotes a sampling weight of the spatial location.
            0 indicates no sampling.
            The weight map shape is assumed ``(spatial_dim_0, spatial_dim_1, ..., spatial_dim_n)``.
        n_samples: number of patch samples
        r_state: a random state container

    Returns:
        a list of `n_samples` N-D integers representing the spatial sampling location of patches.

    """
    if w is None:
        raise ValueError("w must be an ND array.")
    if r_state is None:
        r_state = np.random.RandomState()
    img_size = np.asarray(w.shape, dtype=int)
    win_size = np.asarray(fall_back_tuple(spatial_size, img_size), dtype=int)

    s = tuple(slice(w // 2, m - w + w // 2) if m > w else slice(m // 2, m // 2 + 1) for w, m in zip(win_size, img_size))
    v = w[s]  # weight map in the 'valid' mode
    v_size = v.shape
    v = v.ravel()
    if np.any(v < 0):
        v -= np.min(v)  # shifting to non-negative
    v = v.cumsum()
    if not v[-1] or not np.isfinite(v[-1]) or v[-1] < 0:  # uniform sampling
        idx = r_state.randint(0, len(v), size=n_samples)
    else:
        idx = v.searchsorted(r_state.random(n_samples) * v[-1], side="right")
    # compensate 'valid' mode
    diff = np.minimum(win_size, img_size) // 2
    return [np.unravel_index(i, v_size) + diff for i in np.asarray(idx, dtype=int)]


def generate_pos_neg_label_crop_centers(
    spatial_size: Union[Sequence[int], int],
    num_samples: int,
    pos_ratio: float,
    label_spatial_shape: Sequence[int],
    fg_indices: np.ndarray,
    bg_indices: np.ndarray,
    rand_state: np.random.RandomState = np.random,
) -> List[List[np.ndarray]]:
    """
    Generate valid sample locations based on the label with option for specifying foreground ratio
    Valid: samples sitting entirely within image, expected input shape: [C, H, W, D] or [C, H, W]

    Args:
        spatial_size: spatial size of the ROIs to be sampled.
        num_samples: total sample centers to be generated.
        pos_ratio: ratio of total locations generated that have center being foreground.
        label_spatial_shape: spatial shape of the original label data to unravel selected centers.
        fg_indices: pre-computed foreground indices in 1 dimension.
        bg_indices: pre-computed background indices in 1 dimension.
        rand_state: numpy randomState object to align with other modules.

    Raises:
        ValueError: When the proposed roi is larger than the image.
        ValueError: When the foreground and background indices lengths are 0.

    """
    spatial_size = fall_back_tuple(spatial_size, default=label_spatial_shape)
    if not (np.subtract(label_spatial_shape, spatial_size) >= 0).all():
        raise ValueError("The proposed roi is larger than the image.")

    # Select subregion to assure valid roi
    valid_start = np.floor_divide(spatial_size, 2)
    # add 1 for random
    valid_end = np.subtract(label_spatial_shape + np.array(1), spatial_size / np.array(2)).astype(np.uint16)
    # int generation to have full range on upper side, but subtract unfloored size/2 to prevent rounded range
    # from being too high
    for i in range(len(valid_start)):  # need this because np.random.randint does not work with same start and end
        if valid_start[i] == valid_end[i]:
            valid_end[i] += 1

    def _correct_centers(
        center_ori: List[np.ndarray], valid_start: np.ndarray, valid_end: np.ndarray
    ) -> List[np.ndarray]:
        for i, c in enumerate(center_ori):
            center_i = c
            if c < valid_start[i]:
                center_i = valid_start[i]
            if c >= valid_end[i]:
                center_i = valid_end[i] - 1
            center_ori[i] = center_i
        return center_ori

    centers = []
    fg_indices, bg_indices = np.asarray(fg_indices), np.asarray(bg_indices)
    if fg_indices.size == 0 and bg_indices.size == 0:
        raise ValueError("No sampling location available.")

    if fg_indices.size == 0 or bg_indices.size == 0:
        warnings.warn(
            f"N foreground {len(fg_indices)}, N  background {len(bg_indices)},"
            "unable to generate class balanced samples."
        )
        pos_ratio = 0 if fg_indices.size == 0 else 1

    for _ in range(num_samples):
        indices_to_use = fg_indices if rand_state.rand() < pos_ratio else bg_indices
        random_int = rand_state.randint(len(indices_to_use))
        center = np.unravel_index(indices_to_use[random_int], label_spatial_shape)
        # shift center to range of valid centers
        center_ori = list(center)
        centers.append(_correct_centers(center_ori, valid_start, valid_end))

    return centers


def create_grid(
    spatial_size: Sequence[int],
    spacing: Optional[Sequence[float]] = None,
    homogeneous: bool = True,
    dtype: DtypeLike = float,
):
    """
    compute a `spatial_size` mesh.

    Args:
        spatial_size: spatial size of the grid.
        spacing: same len as ``spatial_size``, defaults to 1.0 (dense grid).
        homogeneous: whether to make homogeneous coordinates.
        dtype: output grid data type.
    """
    spacing = spacing or tuple(1.0 for _ in spatial_size)
    ranges = [np.linspace(-(d - 1.0) / 2.0 * s, (d - 1.0) / 2.0 * s, int(d)) for d, s in zip(spatial_size, spacing)]
    coords = np.asarray(np.meshgrid(*ranges, indexing="ij"), dtype=dtype)
    if not homogeneous:
        return coords
    return np.concatenate([coords, np.ones_like(coords[:1])])


def create_control_grid(
    spatial_shape: Sequence[int], spacing: Sequence[float], homogeneous: bool = True, dtype: DtypeLike = float
):
    """
    control grid with two additional point in each direction
    """
    grid_shape = []
    for d, s in zip(spatial_shape, spacing):
        d = int(d)
        if d % 2 == 0:
            grid_shape.append(np.ceil((d - 1.0) / (2.0 * s) + 0.5) * 2.0 + 2.0)
        else:
            grid_shape.append(np.ceil((d - 1.0) / (2.0 * s)) * 2.0 + 3.0)
    return create_grid(grid_shape, spacing, homogeneous, dtype)


def create_rotate(spatial_dims: int, radians: Union[Sequence[float], float]) -> np.ndarray:
    """
    create a 2D or 3D rotation matrix

    Args:
        spatial_dims: {``2``, ``3``} spatial rank
        radians: rotation radians
            when spatial_dims == 3, the `radians` sequence corresponds to
            rotation in the 1st, 2nd, and 3rd dim respectively.

    Raises:
        ValueError: When ``radians`` is empty.
        ValueError: When ``spatial_dims`` is not one of [2, 3].

    """
    radians = ensure_tuple(radians)
    if spatial_dims == 2:
        if len(radians) >= 1:
            sin_, cos_ = np.sin(radians[0]), np.cos(radians[0])
            return np.array([[cos_, -sin_, 0.0], [sin_, cos_, 0.0], [0.0, 0.0, 1.0]])
        raise ValueError("radians must be non empty.")

    if spatial_dims == 3:
        affine = None
        if len(radians) >= 1:
            sin_, cos_ = np.sin(radians[0]), np.cos(radians[0])
            affine = np.array(
                [[1.0, 0.0, 0.0, 0.0], [0.0, cos_, -sin_, 0.0], [0.0, sin_, cos_, 0.0], [0.0, 0.0, 0.0, 1.0]]
            )
        if len(radians) >= 2:
            sin_, cos_ = np.sin(radians[1]), np.cos(radians[1])
            if affine is None:
                raise ValueError("Affine should be a matrix.")
            affine = affine @ np.array(
                [[cos_, 0.0, sin_, 0.0], [0.0, 1.0, 0.0, 0.0], [-sin_, 0.0, cos_, 0.0], [0.0, 0.0, 0.0, 1.0]]
            )
        if len(radians) >= 3:
            sin_, cos_ = np.sin(radians[2]), np.cos(radians[2])
            if affine is None:
                raise ValueError("Affine should be a matrix.")
            affine = affine @ np.array(
                [[cos_, -sin_, 0.0, 0.0], [sin_, cos_, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
            )
        if affine is None:
            raise ValueError("radians must be non empty.")
        return affine

    raise ValueError(f"Unsupported spatial_dims: {spatial_dims}, available options are [2, 3].")


def create_shear(spatial_dims: int, coefs: Union[Sequence[float], float]) -> np.ndarray:
    """
    create a shearing matrix

    Args:
        spatial_dims: spatial rank
        coefs: shearing factors, defaults to 0.

    Raises:
        NotImplementedError: When ``spatial_dims`` is not one of [2, 3].

    """
    if spatial_dims == 2:
        coefs = ensure_tuple_size(coefs, dim=2, pad_val=0.0)
        return np.array([[1, coefs[0], 0.0], [coefs[1], 1.0, 0.0], [0.0, 0.0, 1.0]])
    if spatial_dims == 3:
        coefs = ensure_tuple_size(coefs, dim=6, pad_val=0.0)
        return np.array(
            [
                [1.0, coefs[0], coefs[1], 0.0],
                [coefs[2], 1.0, coefs[3], 0.0],
                [coefs[4], coefs[5], 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
    raise NotImplementedError("Currently only spatial_dims in [2, 3] are supported.")


def create_scale(spatial_dims: int, scaling_factor: Union[Sequence[float], float]):
    """
    create a scaling matrix

    Args:
        spatial_dims: spatial rank
        scaling_factor: scaling factors, defaults to 1.
    """
    scaling_factor = ensure_tuple_size(scaling_factor, dim=spatial_dims, pad_val=1.0)
    return np.diag(scaling_factor[:spatial_dims] + (1.0,))


def create_translate(spatial_dims: int, shift: Union[Sequence[float], float]) -> np.ndarray:
    """
    create a translation matrix

    Args:
        spatial_dims: spatial rank
        shift: translate factors, defaults to 0.
    """
    shift = ensure_tuple(shift)
    affine = np.eye(spatial_dims + 1)
    for i, a in enumerate(shift[:spatial_dims]):
        affine[i, spatial_dims] = a
    return np.asarray(affine)


def generate_spatial_bounding_box(
    img: np.ndarray,
    select_fn: Callable = lambda x: x > 0,
    channel_indices: Optional[IndexSelection] = None,
    margin: Union[Sequence[int], int] = 0,
) -> Tuple[List[int], List[int]]:
    """
    generate the spatial bounding box of foreground in the image with start-end positions.
    Users can define arbitrary function to select expected foreground from the whole image or specified channels.
    And it can also add margin to every dim of the bounding box.
    The output format of the coordinates is:

        [1st_spatial_dim_start, 2nd_spatial_dim_start, ..., Nth_spatial_dim_start],
        [1st_spatial_dim_end, 2nd_spatial_dim_end, ..., Nth_spatial_dim_end]

    The bounding boxes edges are aligned with the input image edges.
    This function returns [-1, -1, ...], [-1, -1, ...] if there's no positive intensity.

    Args:
        img: source image to generate bounding box from.
        select_fn: function to select expected foreground, default is to select values > 0.
        channel_indices: if defined, select foreground only on the specified channels
            of image. if None, select foreground on the whole image.
        margin: add margin value to spatial dims of the bounding box, if only 1 value provided, use it for all dims.
    """
    data = img[list(ensure_tuple(channel_indices))] if channel_indices is not None else img
    data = np.any(select_fn(data), axis=0)
    ndim = len(data.shape)
    margin = ensure_tuple_rep(margin, ndim)
    for m in margin:
        if m < 0:
            raise ValueError("margin value should not be negative number.")

    box_start = [0] * ndim
    box_end = [0] * ndim

    for di, ax in enumerate(itertools.combinations(reversed(range(ndim)), ndim - 1)):
        dt = data.any(axis=ax)
        if not np.any(dt):
            return [-1] * ndim, [-1] * ndim

        min_d = max(np.argmax(dt) - margin[di], 0)
        max_d = max(data.shape[di] - max(np.argmax(dt[::-1]) - margin[di], 0), min_d + 1)
        box_start[di], box_end[di] = min_d, max_d

    return box_start, box_end


def get_largest_connected_component_mask(img: torch.Tensor, connectivity: Optional[int] = None) -> torch.Tensor:
    """
    Gets the largest connected component mask of an image.

    Args:
        img: Image to get largest connected component from. Shape is (batch_size, spatial_dim1 [, spatial_dim2, ...])
        connectivity: Maximum number of orthogonal hops to consider a pixel/voxel as a neighbor.
            Accepted values are ranging from  1 to input.ndim. If ``None``, a full
            connectivity of ``input.ndim`` is used.
    """
    img_arr = img.detach().cpu().numpy()
    largest_cc = np.zeros(shape=img_arr.shape, dtype=img_arr.dtype)
    for i, item in enumerate(img_arr):
        item = measure.label(item, connectivity=connectivity)
        if item.max() != 0:
            largest_cc[i, ...] = item == (np.argmax(np.bincount(item.flat)[1:]) + 1)
    return torch.as_tensor(largest_cc, device=img.device)


def get_extreme_points(
    img: np.ndarray, rand_state: np.random.RandomState = np.random, background: int = 0, pert: float = 0.0
) -> List[Tuple[int, ...]]:
    """
    Generate extreme points from an image. These are used to generate initial segmentation
    for annotation models. An optional perturbation can be passed to simulate user clicks.

    Args:
        img:
            Image to generate extreme points from. Expected Shape is ``(spatial_dim1, [, spatial_dim2, ...])``.
        rand_state: `np.random.RandomState` object used to select random indices.
        background: Value to be consider as background, defaults to 0.
        pert: Random perturbation amount to add to the points, defaults to 0.0.

    Returns:
        A list of extreme points, its length is equal to 2 * spatial dimension of input image.
        The output format of the coordinates is:

        [1st_spatial_dim_min, 1st_spatial_dim_max, 2nd_spatial_dim_min, ..., Nth_spatial_dim_max]

    Raises:
        ValueError: When the input image does not have any foreground pixel.
    """
    indices = np.where(img != background)
    if np.size(indices[0]) == 0:
        raise ValueError("get_extreme_points: no foreground object in mask!")

    def _get_point(val, dim):
        """
        Select one of the indices within slice containing val.

        Args:
            val : value for comparison
            dim : dimension in which to look for value
        """
        idx = rand_state.choice(np.where(indices[dim] == val)[0])
        pt = []
        for j in range(img.ndim):
            # add +- pert to each dimension
            val = int(indices[j][idx] + 2.0 * pert * (rand_state.rand() - 0.5))
            val = max(val, 0)
            val = min(val, img.shape[j] - 1)
            pt.append(val)
        return pt

    points = []
    for i in range(img.ndim):
        points.append(tuple(_get_point(np.min(indices[i][...]), i)))
        points.append(tuple(_get_point(np.max(indices[i][...]), i)))

    return points


def extreme_points_to_image(
    points: List[Tuple[int, ...]],
    label: np.ndarray,
    sigma: Union[Sequence[float], float, Sequence[torch.Tensor], torch.Tensor] = 0.0,
    rescale_min: float = -1.0,
    rescale_max: float = 1.0,
):
    """
    Please refer to :py:class:`monai.transforms.AddExtremePointsChannel` for the usage.

    Applies a gaussian filter to the extreme points image. Then the pixel values in points image are rescaled
    to range [rescale_min, rescale_max].

    Args:
        points: Extreme points of the object/organ.
        label: label image to get extreme points from. Shape must be
            (1, spatial_dim1, [, spatial_dim2, ...]). Doesn't support one-hot labels.
        sigma: if a list of values, must match the count of spatial dimensions of input data,
            and apply every value in the list to 1 spatial dimension. if only 1 value provided,
            use it for all spatial dimensions.
        rescale_min: minimum value of output data.
        rescale_max: maximum value of output data.
    """
    # points to image
    points_image = torch.zeros(label.shape[1:], dtype=torch.float)
    for p in points:
        points_image[p] = 1.0

    # add channel and add batch
    points_image = points_image.unsqueeze(0).unsqueeze(0)
    gaussian_filter = GaussianFilter(label.ndim - 1, sigma=sigma)
    points_image = gaussian_filter(points_image).squeeze(0).detach().numpy()

    # rescale the points image to [rescale_min, rescale_max]
    min_intensity = np.min(points_image)
    max_intensity = np.max(points_image)
    points_image = (points_image - min_intensity) / (max_intensity - min_intensity)
    points_image = points_image * (rescale_max - rescale_min) + rescale_min
    return points_image


def map_spatial_axes(
    img_ndim: int,
    spatial_axes: Optional[Union[Sequence[int], int]] = None,
    channel_first: bool = True,
) -> List[int]:
    """
    Utility to map the spatial axes to real axes in channel first/last shape.
    For example:
    If `channel_first` is True, and `img` has 3 spatial dims, map spatial axes to real axes as below:
    None -> [1, 2, 3]
    [0, 1] -> [1, 2]
    [0, -1] -> [1, -1]
    If `channel_first` is False, and `img` has 3 spatial dims, map spatial axes to real axes as below:
    None -> [0, 1, 2]
    [0, 1] -> [0, 1]
    [0, -1] -> [0, -2]

    Args:
        img_ndim: dimension number of the target image.
        spatial_axes: spatial axes to be converted, default is None.
            The default `None` will convert to all the spatial axes of the image.
            If axis is negative it counts from the last to the first axis.
            If axis is a tuple of ints.
        channel_first: the image data is channel first or channel last, default to channel first.

    """
    if spatial_axes is None:
        spatial_axes_ = list(range(1, img_ndim) if channel_first else range(0, img_ndim - 1))
    else:
        spatial_axes_ = []
        for a in ensure_tuple(spatial_axes):
            if channel_first:
                spatial_axes_.append(a if a < 0 else a + 1)
            else:
                spatial_axes_.append(a - 1 if a < 0 else a)

    return spatial_axes_


@contextmanager
def allow_missing_keys_mode(transform: Union[MapTransform, Compose, Tuple[MapTransform], Tuple[Compose]]):
    """Temporarily set all MapTransforms to not throw an error if keys are missing. After, revert to original states.

    Args:
        transform: either MapTransform or a Compose

    Example:

    .. code-block:: python

        data = {"image": np.arange(16, dtype=float).reshape(1, 4, 4)}
        t = SpatialPadd(["image", "label"], 10, allow_missing_keys=False)
        _ = t(data)  # would raise exception
        with allow_missing_keys_mode(t):
            _ = t(data)  # OK!
    """
    # If given a sequence of transforms, Compose them to get a single list
    if issequenceiterable(transform):
        transform = Compose(transform)

    # Get list of MapTransforms
    transforms = []
    if isinstance(transform, MapTransform):
        transforms = [transform]
    elif isinstance(transform, Compose):
        # Only keep contained MapTransforms
        transforms = [t for t in transform.flatten().transforms if isinstance(t, MapTransform)]
    if len(transforms) == 0:
        raise TypeError(
            "allow_missing_keys_mode expects either MapTransform(s) or Compose(s) containing MapTransform(s)"
        )

    # Get the state of each `allow_missing_keys`
    orig_states = [t.allow_missing_keys for t in transforms]

    try:
        # Set all to True
        for t in transforms:
            t.allow_missing_keys = True
        yield
    finally:
        # Revert
        for t, o_s in zip(transforms, orig_states):
            t.allow_missing_keys = o_s


def convert_inverse_interp_mode(trans_info: List, mode: str = "nearest", align_corners: Optional[bool] = None):
    """
    Change the interpolation mode when inverting spatial transforms, default to "nearest".
    This function modifies trans_info's `InverseKeys.EXTRA_INFO`.

    See also: :py:class:`monai.transform.inverse.InvertibleTransform`

    Args:
        trans_info: transforms inverse information list, contains context of every invertible transform.
        mode: target interpolation mode to convert, default to "nearest" as it's usually used to save the mode output.
        align_corners: target align corner value in PyTorch interpolation API, need to align with the `mode`.

    """
    interp_modes = [i.value for i in InterpolateMode] + [i.value for i in GridSampleMode]

    # set to string for DataLoader collation
    align_corners_ = "none" if align_corners is None else align_corners

    for item in ensure_tuple(trans_info):
        if InverseKeys.EXTRA_INFO in item:
            orig_mode = item[InverseKeys.EXTRA_INFO].get("mode", None)
            if orig_mode is not None:
                if orig_mode[0] in interp_modes:
                    item[InverseKeys.EXTRA_INFO]["mode"] = [mode for _ in range(len(mode))]
                elif orig_mode in interp_modes:
                    item[InverseKeys.EXTRA_INFO]["mode"] = mode
            if "align_corners" in item[InverseKeys.EXTRA_INFO]:
                if issequenceiterable(item[InverseKeys.EXTRA_INFO]["align_corners"]):
                    item[InverseKeys.EXTRA_INFO]["align_corners"] = [align_corners_ for _ in range(len(mode))]
                else:
                    item[InverseKeys.EXTRA_INFO]["align_corners"] = align_corners_
    return trans_info

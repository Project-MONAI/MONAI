# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
import warnings
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from monai.config import IndexSelection
from monai.utils import ensure_tuple, ensure_tuple_rep, ensure_tuple_size, fall_back_tuple, min_version, optional_import

measure, _ = optional_import("skimage.measure", "0.14.2", min_version)


def rand_choice(prob: float = 0.5) -> bool:
    """
    Returns True if a randomly chosen number is less than or equal to `prob`, by default this is a 50/50 chance.
    """
    return bool(random.random() <= prob)


def img_bounds(img: np.ndarray) -> np.ndarray:
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

    if np.any(img[:, :margin, :]) or np.any(img[:, -margin:, :]):
        return False

    return True


def rescale_array(
    arr: np.ndarray, minv: float = 0.0, maxv: float = 1.0, dtype: Optional[np.dtype] = np.float32
) -> np.ndarray:
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
    arr: np.ndarray, minv: float = 0.0, maxv: float = 1.0, dtype: np.dtype = np.float32
) -> np.ndarray:
    """
    Rescale each array slice along the first dimension of `arr` independently.
    """
    out: np.ndarray = np.zeros(arr.shape, dtype)
    for i in range(arr.shape[0]):
        out[i] = rescale_array(arr[i], minv, maxv, dtype)

    return out


def rescale_array_int_max(arr: np.ndarray, dtype: np.dtype = np.uint16) -> np.ndarray:
    """
    Rescale the array `arr` to be between the minimum and maximum values of the type `dtype`.
    """
    info: np.iinfo = np.iinfo(dtype)
    return rescale_array(arr, info.min, info.max).astype(dtype)


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


def resize_center(
    img: np.ndarray, *resize_dims: Optional[int], fill_value: float = 0.0, inplace: bool = True
) -> np.ndarray:
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
        dest = np.full(resize_dims, fill_value, img.dtype)
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
    img_size = np.asarray(w.shape, dtype=np.int)
    win_size = np.asarray(fall_back_tuple(spatial_size, img_size), dtype=np.int)

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
    centers = [np.unravel_index(i, v_size) + diff for i in np.asarray(idx, dtype=np.int)]
    return centers


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

    if not len(fg_indices) or not len(bg_indices):
        if not len(fg_indices) and not len(bg_indices):
            raise ValueError("No sampling location available.")
        warnings.warn(
            f"N foreground {len(fg_indices)}, N  background {len(bg_indices)},"
            "unable to generate class balanced samples."
        )
        pos_ratio = 0 if not len(fg_indices) else 1

    for _ in range(num_samples):
        indices_to_use = fg_indices if rand_state.rand() < pos_ratio else bg_indices
        random_int = rand_state.randint(len(indices_to_use))
        center = np.unravel_index(indices_to_use[random_int], label_spatial_shape)
        # shift center to range of valid centers
        center_ori = list(center)
        centers.append(_correct_centers(center_ori, valid_start, valid_end))

    return centers


def apply_transform(transform: Callable, data, map_items: bool = True):
    """
    Transform `data` with `transform`.
    If `data` is a list or tuple and `map_data` is True, each item of `data` will be transformed
    and this method returns a list of outcomes.
    otherwise transform will be applied once with `data` as the argument.

    Args:
        transform: a callable to be used to transform `data`
        data: an object to be transformed.
        map_items: whether to apply transform to each item in `data`,
            if `data` is a list or tuple. Defaults to True.

    Raises:
        Exception: When ``transform`` raises an exception.

    """
    try:
        if isinstance(data, (list, tuple)) and map_items:
            return [transform(item) for item in data]
        return transform(data)
    except Exception as e:
        raise RuntimeError(f"applying transform {transform}") from e


def create_grid(
    spatial_size: Sequence[int],
    spacing: Optional[Sequence[float]] = None,
    homogeneous: bool = True,
    dtype: np.dtype = float,
) -> np.ndarray:
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
    spatial_shape: Sequence[int], spacing: Sequence[float], homogeneous: bool = True, dtype: np.dtype = float
) -> np.ndarray:
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
            affine = affine @ np.array(
                [[cos_, 0.0, sin_, 0.0], [0.0, 1.0, 0.0, 0.0], [-sin_, 0.0, cos_, 0.0], [0.0, 0.0, 0.0, 1.0]]
            )
        if len(radians) >= 3:
            sin_, cos_ = np.sin(radians[2]), np.cos(radians[2])
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


def create_scale(spatial_dims: int, scaling_factor: Union[Sequence[float], float]) -> np.ndarray:
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
    return affine


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

    Args:
        img: source image to generate bounding box from.
        select_fn: function to select expected foreground, default is to select values > 0.
        channel_indices: if defined, select foreground only on the specified channels
            of image. if None, select foreground on the whole image.
        margin: add margin value to spatial dims of the bounding box, if only 1 value provided, use it for all dims.
    """
    data = img[[*(ensure_tuple(channel_indices))]] if channel_indices is not None else img
    data = np.any(select_fn(data), axis=0)
    nonzero_idx = np.nonzero(data)
    margin = ensure_tuple_rep(margin, data.ndim)

    box_start = list()
    box_end = list()
    for i in range(data.ndim):
        assert len(nonzero_idx[i]) > 0, f"did not find nonzero index at spatial dim {i}"
        box_start.append(max(0, np.min(nonzero_idx[i]) - margin[i]))
        box_end.append(min(data.shape[i], np.max(nonzero_idx[i]) + margin[i] + 1))
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

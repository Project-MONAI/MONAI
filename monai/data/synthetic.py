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

import numpy as np

from monai.transforms.utils import rescale_array

__all__ = ["create_test_image_2d", "create_test_image_3d"]


def create_test_image_2d(
    height: int,
    width: int,
    num_objs: int = 12,
    rad_max: int = 30,
    rad_min: int = 5,
    noise_max: float = 0.0,
    num_seg_classes: int = 5,
    channel_dim: int | None = None,
    random_state: np.random.RandomState | None = None,
    return_instance_id: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return a noisy 2D image with `num_objs` circles and a 2D mask image. The maximum and minimum radii of the circles
    are given as `rad_max` and `rad_min`. The mask will have `num_seg_classes` number of classes for segmentations labeled
    sequentially from 1, plus a background class represented as 0. If `noise_max` is greater than 0 then noise will be
    added to the image taken from the uniform distribution on range `[0,noise_max)`. If `channel_dim` is None, will create
    an image without channel dimension, otherwise create an image with channel dimension as first dim or last dim.

    Args:
        height: height of the image. The value should be larger than `2 * rad_max`.
        width: width of the image. The value should be larger than `2 * rad_max`.
        num_objs: number of circles to generate. Defaults to `12`.
        rad_max: maximum circle radius. Defaults to `30`.
        rad_min: minimum circle radius. Defaults to `5`.
        noise_max: if greater than 0 then noise will be added to the image taken from
            the uniform distribution on range `[0,noise_max)`. Defaults to `0`.
        num_seg_classes: number of classes for segmentations. Defaults to `5`.
        channel_dim: if None, create an image without channel dimension, otherwise create
            an image with channel dimension as first dim or last dim. Defaults to `None`.
        random_state: the random generator to use. Defaults to `np.random`.
        return_instance_id: if True, also return an instance ID mask where every generated
            object is assigned a unique positive integer. Defaults to `False`.

    Returns:
        A tuple of image and segmentation label arrays. If `return_instance_id=True`, also returns
        an instance ID array as the third element.
    """

    if rad_max <= rad_min:
        raise ValueError(f"`rad_min` {rad_min} should be less than `rad_max` {rad_max}.")
    if rad_min < 1:
        raise ValueError(f"`rad_min` {rad_min} should be no less than 1.")
    min_size = min(height, width)
    if min_size <= 2 * rad_max:
        raise ValueError(f"the minimal size {min_size} of the image should be larger than `2 * rad_max` 2x{rad_max}.")

    image = np.zeros((height, width))
    instance_ids = np.zeros((height, width), dtype=np.int32)
    rs: np.random.RandomState = np.random.random.__self__ if random_state is None else random_state  # type: ignore

    for obj_id in range(1, num_objs + 1):
        x = rs.randint(rad_max, height - rad_max)
        y = rs.randint(rad_max, width - rad_max)
        rad = rs.randint(rad_min, rad_max)
        spy, spx = np.ogrid[-x : height - x, -y : width - y]
        circle = (spx * spx + spy * spy) <= rad * rad

        if num_seg_classes > 1:
            image[circle] = np.ceil(rs.random() * num_seg_classes)
        else:
            image[circle] = rs.random() * 0.5 + 0.5
        instance_ids[circle] = obj_id

    labels = np.ceil(image).astype(np.int32, copy=False)

    norm = rs.uniform(0, num_seg_classes * noise_max, size=image.shape)
    noisyimage: np.ndarray = rescale_array(np.maximum(image, norm))  # type: ignore

    if channel_dim is not None:
        if not (isinstance(channel_dim, int) and channel_dim in (-1, 0, 2)):
            raise AssertionError("invalid channel dim.")
        if channel_dim == 0:
            noisyimage = noisyimage[None]
            labels = labels[None]
            instance_ids = instance_ids[None]
        else:
            noisyimage = noisyimage[..., None]
            labels = labels[..., None]
            instance_ids = instance_ids[..., None]

    if return_instance_id:
        return noisyimage, labels, instance_ids
    return noisyimage, labels


def create_test_image_3d(
    height: int,
    width: int,
    depth: int,
    num_objs: int = 12,
    rad_max: int = 30,
    rad_min: int = 5,
    noise_max: float = 0.0,
    num_seg_classes: int = 5,
    channel_dim: int | None = None,
    random_state: np.random.RandomState | None = None,
    return_instance_id: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return a noisy 3D image and segmentation.

    Args:
        height: height of the image. The value should be larger than `2 * rad_max`.
        width: width of the image. The value should be larger than `2 * rad_max`.
        depth: depth of the image. The value should be larger than `2 * rad_max`.
        num_objs: number of circles to generate. Defaults to `12`.
        rad_max: maximum circle radius. Defaults to `30`.
        rad_min: minimum circle radius. Defaults to `5`.
        noise_max: if greater than 0 then noise will be added to the image taken from
            the uniform distribution on range `[0,noise_max)`. Defaults to `0`.
        num_seg_classes: number of classes for segmentations. Defaults to `5`.
        channel_dim: if None, create an image without channel dimension, otherwise create
            an image with channel dimension as first dim or last dim. Defaults to `None`.
        random_state: the random generator to use. Defaults to `np.random`.
        return_instance_id: if True, also return an instance ID mask where every generated
            object is assigned a unique positive integer. Defaults to `False`.

    Returns:
        A tuple of image and segmentation label arrays. If `return_instance_id=True`, also returns
        an instance ID array as the third element.

    See also:
        :py:meth:`~create_test_image_2d`
    """

    if rad_max <= rad_min:
        raise ValueError(f"`rad_min` {rad_min} should be less than `rad_max` {rad_max}.")
    if rad_min < 1:
        raise ValueError("f`rad_min` {rad_min} should be no less than 1.")
    min_size = min(height, width, depth)
    if min_size <= 2 * rad_max:
        raise ValueError(f"the minimal size {min_size} of the image should be larger than `2 * rad_max` 2x{rad_max}.")

    image = np.zeros((height, width, depth))
    instance_ids = np.zeros((height, width, depth), dtype=np.int32)
    rs: np.random.RandomState = np.random.random.__self__ if random_state is None else random_state  # type: ignore

    for obj_id in range(1, num_objs + 1):
        x = rs.randint(rad_max, height - rad_max)
        y = rs.randint(rad_max, width - rad_max)
        z = rs.randint(rad_max, depth - rad_max)
        rad = rs.randint(rad_min, rad_max)
        spy, spx, spz = np.ogrid[-x : height - x, -y : width - y, -z : depth - z]
        circle = (spx * spx + spy * spy + spz * spz) <= rad * rad

        if num_seg_classes > 1:
            image[circle] = np.ceil(rs.random() * num_seg_classes)
        else:
            image[circle] = rs.random() * 0.5 + 0.5
        instance_ids[circle] = obj_id

    labels = np.ceil(image).astype(np.int32, copy=False)

    norm = rs.uniform(0, num_seg_classes * noise_max, size=image.shape)
    noisyimage: np.ndarray = rescale_array(np.maximum(image, norm))  # type: ignore

    if channel_dim is not None:
        if not (isinstance(channel_dim, int) and channel_dim in (-1, 0, 3)):
            raise AssertionError("invalid channel dim.")
        if channel_dim == 0:
            noisyimage = noisyimage[None]
            labels = labels[None]
            instance_ids = instance_ids[None]
        else:
            noisyimage = noisyimage[..., None]
            labels = labels[..., None]
            instance_ids = instance_ids[..., None]

    if return_instance_id:
        return noisyimage, labels, instance_ids
    return noisyimage, labels

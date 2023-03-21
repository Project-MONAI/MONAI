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

from typing import Optional, Sequence, Union

import torch
from monai.config import DtypeLike

from monai.data.meta_obj import get_track_meta
from monai.transforms.utils import create_translate
from monai.transforms.lazy.functional import extents_from_shape, shape_from_extents, lazily_apply_op
# from monai.transforms.meta_matrix import MatrixFactory
from monai.transforms.lazy.utils import MetaMatrix
from monai.transforms.spatial.functional import get_input_shape_and_dtype, transform_shape
from monai.utils import GridSamplePadMode, NumpyPadMode, convert_to_tensor, LazyAttr


def transform_from_slices(input_shape, slices):
    input_ndim = len(input_shape) - 1
    img_centers = [i / 2 for i in input_shape[1:]]
    slice_centers = [(s.stop + s.start) / 2 for s in slices]
    deltas = [s - i for i, s in zip(img_centers, slice_centers)]
    transform = create_translate(input_ndim, deltas)
    output_shape = transform_shape([input_shape[0]] + [s.stop - s.start for s in slices], transform)

    return transform, output_shape


def croppad(
        img: torch.Tensor,
        slices: Union[Sequence[slice], slice],
        padding_mode: Optional[Union[GridSamplePadMode, str]] = NumpyPadMode.EDGE,
        shape_override: Sequence | None = None,
        dtype_override: DtypeLike | torch.dtype | None = None,
        lazy_evaluation: Optional[bool] = True
):
    img_ = convert_to_tensor(img, track_meta=get_track_meta())

    input_shape, input_dtype = get_input_shape_and_dtype(shape_override, dtype_override, img_)

    input_ndim = len(input_shape) - 1

    if len(slices) != input_ndim:
        raise ValueError(f"'slices' length {len(slices)} must be equal to 'img' "
                         f"spatial dimensions of {input_ndim}")

    # img_centers = [i / 2 for i in input_shape[1:]]
    # slice_centers = [(s.stop + s.start) / 2 for s in slices]
    # deltas = [s - i for i, s in zip(img_centers, slice_centers)]
    # transform = create_translate(input_ndim, deltas)
    # output_shape = transform_shape([input_shape[0]] + [s.stop - s.start for s in slices], transform)
    transform, output_shape = transform_from_slices(input_shape, slices)
    # im_extents = extents_from_shape([input_shape[0]] + [s.stop - s.start for s in slices])
    # im_extents = [transform @ e for e in im_extents]
    # shape_override_ = shape_from_extents(input_shape, im_extents)

    metadata = {
        "op": "croppad",
        "slices": slices,
        LazyAttr.PADDING_MODE: padding_mode,
        LazyAttr.IN_SHAPE: input_shape,
        LazyAttr.IN_DTYPE: input_dtype,
        LazyAttr.OUT_DTYPE: input_dtype,
        LazyAttr.OUT_SHAPE: output_shape
    }
    return lazily_apply_op(img_, MetaMatrix(transform, metadata), lazy_evaluation)


def pad(
        img: torch.Tensor,
        padding: Sequence[tuple[int, int]] | tuple[int, int],
        padding_mode: str | None = "border",
        value: int | float = 0,
        shape_override: Sequence[int] | None = None,
        dtype_override: DtypeLike | torch.dtype | None = None,
        lazy_evaluation: Optional[bool] = True
):
    img_ = convert_to_tensor(img, track_meta=get_track_meta())

    input_shape, input_dtype = get_input_shape_and_dtype(shape_override, dtype_override, img_)

    input_ndim = len(input_shape) - 1

    slices = list()
    for d in range(input_ndim):
        if d < len(padding):
            slices.append(slice(-padding[d][0], input_shape[d+1] + padding[d][1]))
        else:
            slices.append(slice(0, input_shape[d+1]))

    transform, output_shape = transform_from_slices(input_shape, slices)

    # TODO: maybe don't include value unless it is needed for the padding_mode
    metadata = {
        "op": "croppad",
        "slices": slices,
        LazyAttr.PADDING_MODE: padding_mode,
        LazyAttr.IN_SHAPE: input_shape,
        LazyAttr.IN_DTYPE: input_dtype,
        LazyAttr.OUT_DTYPE: input_dtype,
        LazyAttr.OUT_SHAPE: output_shape,
        "value": value
    }
    return lazily_apply_op(img_, MetaMatrix(transform, metadata), lazy_evaluation)

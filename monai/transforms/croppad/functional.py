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

from typing import Optional, Sequence, Union

import torch

from monai.data.meta_obj import get_track_meta
from monai.transforms import create_translate
from monai.transforms.lazy.functional import extents_from_shape, shape_from_extents, lazily_apply_op
# from monai.transforms.meta_matrix import MatrixFactory
from monai.transforms.lazy.utils import MetaMatrix
from monai.transforms.spatial.functional import get_input_shape_and_dtype
from monai.utils import GridSamplePadMode, NumpyPadMode, convert_to_tensor, LazyAttr


def croppad(
        img: torch.Tensor,
        slices: Union[Sequence[slice], slice],
        padding_mode: Optional[Union[GridSamplePadMode, str]] = NumpyPadMode.EDGE,
        shape_override: Optional[Sequence] = None,
        dtype_override: Optional[bool] = None,
        lazy_evaluation: Optional[bool] = True
):
    img_ = convert_to_tensor(img, track_meta=get_track_meta())

    input_shape, input_dtype = get_input_shape_and_dtype(shape_override, dtype_override, img_)

    input_ndim = len(input_shape) - 1

    if len(slices) != input_ndim:
        raise ValueError(f"'slices' length {len(slices)} must be equal to 'img' "
                         f"spatial dimensions of {input_ndim}")

    img_centers = [i / 2 for i in input_shape[1:]]
    slice_centers = [(s.stop + s.start) / 2 for s in slices]
    deltas = [s - i for i, s in zip(img_centers, slice_centers)]
    transform = create_translate(input_ndim, deltas)
    im_extents = extents_from_shape([input_shape[0]] + [s.stop - s.start for s in slices])
    im_extents = [transform @ e for e in im_extents]
    shape_override_ = shape_from_extents(input_shape, im_extents)

    metadata = {
        "op": "croppad",
        "slices": slices,
        LazyAttr.PADDING_MODE: padding_mode,
        LazyAttr.IN_SHAPE: input_shape,
        LazyAttr.IN_DTYPE: input_dtype,
        LazyAttr.OUT_DTYPE: input_dtype,
        LazyAttr.OUT_SHAPE: shape_override_
    }
    return lazily_apply_op(img_, MetaMatrix(transform, metadata), lazy_evaluation)

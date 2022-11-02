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
from monai.transforms.lazy.functional import extents_from_shape, shape_from_extents
from monai.transforms.meta_matrix import MatrixFactory
from monai.utils import GridSamplePadMode, NumpyPadMode, convert_to_tensor


def croppad(
        img: torch.Tensor,
        slices: Union[Sequence[slice], slice],
        padding_mode: Optional[Union[GridSamplePadMode, str]] = NumpyPadMode.EDGE,
        shape_override: Optional[Sequence] = None
):
    img_ = convert_to_tensor(img, track_meta=get_track_meta())
    input_shape = img_.shape if shape_override is None else shape_override
    input_ndim = len(input_shape) - 1
    if len(slices) != input_ndim:
        raise ValueError(f"'slices' length {len(slices)} must be equal to 'img' "
                         f"spatial dimensions of {input_ndim}")

    img_centers = [i / 2 for i in input_shape[1:]]
    slice_centers = [(s.stop + s.start) / 2 for s in slices]
    deltas = [s - i for i, s in zip(img_centers, slice_centers)]
    transform = MatrixFactory.from_tensor(img).translate(deltas).matrix.matrix
    im_extents = extents_from_shape([input_shape[0]] + [s.stop - s.start for s in slices])
    im_extents = [transform @ e for e in im_extents]
    shape_override_ = shape_from_extents(input_shape, im_extents)

    metadata = {
        "slices": slices,
        "padding_mode": padding_mode,
        "dtype": img.dtype,
        "im_extents": im_extents,
        "shape_override": shape_override_
    }
    return img_, transform, metadata

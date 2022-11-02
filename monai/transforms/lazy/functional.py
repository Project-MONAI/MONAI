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

from typing import Sequence, Union

import itertools as it

import numpy as np

import torch

#
# placeholder that will conflict with PR Replacement Apply and Resample #5436
def apply(*args, **kwargs):
    raise NotImplementedError()


# this will conflict with PR Replacement Apply and Resample #5436
def extents_from_shape(shape, dtype=np.float32):
    extents = [[0, shape[i]] for i in range(1, len(shape))]

    extents = it.product(*extents)
    return [np.asarray(e + (1,), dtype=dtype) for e in extents]


def shape_from_extents(
    src_shape: Sequence, extents: Union[Sequence[np.ndarray], Sequence[torch.Tensor], np.ndarray, torch.Tensor]
):
    if isinstance(extents, (list, tuple)):
        if isinstance(extents[0], np.ndarray):
            aextents = np.asarray(extents)
        else:
            aextents = torch.stack(extents)
            aextents = aextents.numpy()
    else:
        if isinstance(extents, np.ndarray):
            aextents = extents
        else:
            aextents = extents.numpy()

    mins = aextents.min(axis=0)
    maxes = aextents.max(axis=0)
    values = np.round(maxes - mins).astype(int)[:-1].tolist()
    return (src_shape[0],) + tuple(values)
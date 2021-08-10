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
"""
A collection of "vanilla" transforms for crop and pad operations acting on batches of data
https://github.com/Project-MONAI/MONAI/wiki/MONAI_Design
"""

from copy import deepcopy
from typing import Any, Dict, Hashable, Union

import numpy as np
import torch

from monai.data.utils import list_data_collate
from monai.transforms.compose import Compose
from monai.transforms.croppad.array import CenterSpatialCrop, SpatialPad
from monai.transforms.inverse import InvertibleTransform
from monai.transforms.utility.array import ToTensor
from monai.utils.enums import InverseKeys, Method, NumpyPadMode

__all__ = [
    "PadListDataCollate",
]


def replace_element(to_replace, batch, idx, key_or_idx):
    # since tuple is immutable we'll have to recreate
    if isinstance(batch[idx], tuple):
        batch_idx_list = list(batch[idx])
        batch_idx_list[key_or_idx] = to_replace
        batch[idx] = tuple(batch_idx_list)
    # else, replace
    else:
        batch[idx][key_or_idx] = to_replace
    return batch


class PadListDataCollate(InvertibleTransform):
    """
    Same as MONAI's ``list_data_collate``, except any tensors are centrally padded to match the shape of the biggest
    tensor in each dimension. This transform is useful if some of the applied transforms generate batch data of
    different sizes.

    This can be used on both list and dictionary data. In the case of the dictionary data, this transform will be added
    to the list of invertible transforms.

    Note that normally, a user won't explicitly use the `__call__` method. Rather this would be passed to the `DataLoader`.
    This means that `__call__` handles data as it comes out of a `DataLoader`, containing batch dimension. However, the
    `inverse` operates on dictionaries containing images of shape `C,H,W,[D]`. This asymmetry is necessary so that we can
    pass the inverse through multiprocessing.

    Args:
        method: padding method (see :py:class:`monai.transforms.SpatialPad`)
        mode: padding mode (see :py:class:`monai.transforms.SpatialPad`)
        np_kwargs: other args for `np.pad` API, note that `np.pad` treats channel dimension as the first dimension.
            more details: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html

    """

    def __init__(
        self,
        method: Union[Method, str] = Method.SYMMETRIC,
        mode: Union[NumpyPadMode, str] = NumpyPadMode.CONSTANT,
        **np_kwargs,
    ) -> None:
        self.method = method
        self.mode = mode
        self.np_kwargs = np_kwargs

    def __call__(self, batch: Any):
        """
        Args:
            batch: batch of data to pad-collate
        """
        # data is either list of dicts or list of lists
        is_list_of_dicts = isinstance(batch[0], dict)
        # loop over items inside of each element in a batch
        for key_or_idx in batch[0].keys() if is_list_of_dicts else range(len(batch[0])):
            # calculate max size of each dimension
            max_shapes = []
            for elem in batch:
                if not isinstance(elem[key_or_idx], (torch.Tensor, np.ndarray)):
                    break
                max_shapes.append(elem[key_or_idx].shape[1:])
            # len > 0 if objects were arrays, else skip as no padding to be done
            if not max_shapes:
                continue
            max_shape = np.array(max_shapes).max(axis=0)
            # If all same size, skip
            if np.all(np.array(max_shapes).min(axis=0) == max_shape):
                continue
            # Do we need to convert output to Tensor?
            output_to_tensor = isinstance(batch[0][key_or_idx], torch.Tensor)

            # Use `SpatialPadd` or `SpatialPad` to match sizes
            # Default params are central padding, padding with 0's
            # If input is dictionary, use the dictionary version so that the transformation is recorded

            padder = SpatialPad(spatial_size=max_shape, method=self.method, mode=self.mode, **self.np_kwargs)
            transform = padder if not output_to_tensor else Compose([padder, ToTensor()])

            for idx, batch_i in enumerate(batch):
                im = batch_i[key_or_idx]
                orig_size = im.shape[1:]
                padded = transform(batch_i[key_or_idx])
                batch = replace_element(padded, batch, idx, key_or_idx)

                # If we have a dictionary of data, append to list
                if is_list_of_dicts:
                    self.push_transform(batch[idx], key_or_idx, orig_size=orig_size)

        # After padding, use default list collator
        return list_data_collate(batch)

    @staticmethod
    def inverse(data: dict) -> Dict[Hashable, np.ndarray]:
        if not isinstance(data, dict):
            raise RuntimeError("Inverse can only currently be applied on dictionaries.")

        d = deepcopy(data)
        for key in d:
            transform_key = str(key) + InverseKeys.KEY_SUFFIX
            if transform_key in d:
                transform = d[transform_key][-1]
                if transform[InverseKeys.CLASS_NAME] == PadListDataCollate.__name__:
                    d[key] = CenterSpatialCrop(transform["orig_size"])(d[key])
                    # remove transform
                    d[transform_key].pop()
        return d

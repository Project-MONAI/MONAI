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
"""
A collection of "vanilla" transforms for crop and pad operations acting on batches of data
https://github.com/Project-MONAI/MONAI/wiki/MONAI_Design
"""

from typing import Any, Dict, Hashable, Mapping

import numpy as np
import torch

from monai.data.meta_tensor import MetaTensor
from monai.data.utils import list_data_collate
from monai.transforms.croppad.array import CenterSpatialCrop, SpatialPad
from monai.transforms.inverse import InvertibleTransform
from monai.utils.enums import Method, PytorchPadMode, TraceKeys

__all__ = ["PadListDataCollate"]


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

    This can be used on both list and dictionary data.
    Note that in the case of the dictionary data, it may add the transform information to the list of invertible transforms
    if input batch have different spatial shape, so need to call static method: `inverse` before inverting other transforms.

    Note that normally, a user won't explicitly use the `__call__` method. Rather this would be passed to the `DataLoader`.
    This means that `__call__` handles data as it comes out of a `DataLoader`, containing batch dimension. However, the
    `inverse` operates on dictionaries containing images of shape `C,H,W,[D]`. This asymmetry is necessary so that we can
    pass the inverse through multiprocessing.

    Args:
        method: padding method (see :py:class:`monai.transforms.SpatialPad`)
        mode: padding mode (see :py:class:`monai.transforms.SpatialPad`)
        kwargs: other arguments for the `np.pad` or `torch.pad` function.
            note that `np.pad` treats channel dimension as the first dimension.

    """

    def __init__(self, method: str = Method.SYMMETRIC, mode: str = PytorchPadMode.CONSTANT, **kwargs) -> None:
        self.method = method
        self.mode = mode
        self.kwargs = kwargs

    def __call__(self, batch: Any):
        """
        Args:
            batch: batch of data to pad-collate
        """
        # data is either list of dicts or list of lists
        is_list_of_dicts = isinstance(batch[0], dict)
        # loop over items inside of each element in a batch
        batch_item = tuple(batch[0].keys()) if is_list_of_dicts else range(len(batch[0]))
        for key_or_idx in batch_item:
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

            # Use `SpatialPad` to match sizes, Default params are central padding, padding with 0's
            padder = SpatialPad(spatial_size=max_shape, method=self.method, mode=self.mode, **self.kwargs)
            for idx, batch_i in enumerate(batch):
                orig_size = batch_i[key_or_idx].shape[1:]
                padded = padder(batch_i[key_or_idx])
                batch = replace_element(padded, batch, idx, key_or_idx)

                # If we have a dictionary of data, append to list
                if is_list_of_dicts:
                    self.push_transform(
                        batch[idx],
                        key_or_idx,
                        orig_size=orig_size,
                        extra_info=self.pop_transform(batch[idx], key_or_idx, check=False),
                    )

        # After padding, use default list collator
        return list_data_collate(batch)

    @staticmethod
    def inverse(data: dict) -> Dict[Hashable, np.ndarray]:
        if not isinstance(data, Mapping):
            raise RuntimeError("Inverse can only currently be applied on dictionaries.")

        d = dict(data)
        for key in d:
            transforms = None
            if isinstance(d[key], MetaTensor):
                transforms = d[key].applied_operations
            else:
                transform_key = InvertibleTransform.trace_key(key)
                if transform_key in d:
                    transforms = d[transform_key]
            if not transforms or not isinstance(transforms[-1], Dict):
                continue
            if transforms[-1].get(TraceKeys.CLASS_NAME) == PadListDataCollate.__name__:
                xform = transforms.pop()
                cropping = CenterSpatialCrop(xform.get(TraceKeys.ORIG_SIZE, -1))
                with cropping.trace_transform(False):
                    d[key] = cropping(d[key])  # fallback to image size
        return d

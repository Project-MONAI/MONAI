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
from typing import Any, Hashable

import torch

from monai.transforms import Compose, OneOf, apply_transform


def get_number_image_type_conversions(transform: Compose, test_data: Any, key: Hashable | None = None) -> int:
    """
    Get the number of times that the data need to be converted (e.g., numpy to torch).
    Conversions between different devices are also counted (e.g., CPU to GPU).

    Args:
        transform: composed transforms to be tested
        test_data: data to be used to count the number of conversions
        key: if using dictionary transforms, this key will be used to check the number of conversions.
    """

    def _get_data(obj, key):
        return obj if key is None else obj[key]

    # if the starting point is a string (e.g., input to LoadImage), start
    # at -1 since we don't want to count the string -> image conversion.
    num_conversions = 0 if not isinstance(_get_data(test_data, key), str) else -1

    tr = transform.flatten().transforms

    if isinstance(transform, OneOf) or any(isinstance(i, OneOf) for i in tr):
        raise RuntimeError("Not compatible with `OneOf`, as the applied transform is deterministically chosen.")

    for _transform in tr:
        prev_data = _get_data(test_data, key)
        prev_type = type(prev_data)
        prev_device = prev_data.device if isinstance(prev_data, torch.Tensor) else None
        test_data = apply_transform(_transform, test_data, transform.map_items, transform.unpack_items)
        # every time the type or device changes, increment the counter
        curr_data = _get_data(test_data, key)
        curr_device = curr_data.device if isinstance(curr_data, torch.Tensor) else None
        if not isinstance(curr_data, prev_type) or curr_device != prev_device:
            num_conversions += 1
    return num_conversions

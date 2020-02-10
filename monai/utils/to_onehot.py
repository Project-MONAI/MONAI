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

import torch
import torch.nn.functional as f


def to_onehot(data, num_classes: int):
    """Utility function to convert PyTorch tensor to One-Hot encoding format.
    The input data should only have 1 channel and the first dim is batch.
    Example shapes: [16, 1, 96, 96], [16, 1, 96, 96, 32].
    And the data values must match "num_classes"， example: num_classes = 10 and values in [0 ... 9].

    Args:
        data (torch.Tensor): target data to convert One-Hot format.
        num_classes (int): the class number in the Tensor data.

    """
    num_dims = data.dim()
    if num_dims < 2 or data.shape[1] != 1:
        raise ValueError('data should have a channel with length equals to one.')

    data = torch.squeeze(data, 1)
    data = f.one_hot(data.long(), num_classes)
    new_axes = [0, -1] + list(range(1, num_dims - 1))
    data = data.permute(*new_axes)
    if not data.is_contiguous():
        return data.contiguous()
    return data

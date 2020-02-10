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


def to_onehot(data, num_classes):
    """Util function to convert PyTorch tensor to One-Hot encoding format.
    The input data should only have 1 channel and the first dim is batch.
    Example shapes: [16, 1, 96, 96], [16, 1, 96, 96, 32].
    And the data values must match "num_classes"， example: num_classes = 10 and values in [0 ... 9].

    Args:
        data (torch.Tensor): target data to convert One-Hot format.
        num_classes (Int): the class number in the Tensor data.

    """
    assert num_classes is not None and type(num_classes) == int, 'must set class number for one-hot.'

    data = torch.squeeze(data, 1)
    data = f.one_hot(data.long(), num_classes)
    num_dims = len(data.shape)
    assert num_dims == 4 or num_dims == 5, 'unsupported input shape.'
    if num_dims == 5:
        data = data.permute(0, 4, 1, 2, 3)
    elif num_dims == 4:
        data = data.permute(0, 3, 1, 2)
    if data.is_contiguous() is False:
        data = data.contiguous()
    return data

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

from abc import abstractmethod
from typing import Sequence, Union

from torch import Tensor, nn

from monai.utils import ensure_tuple_rep
from monai.utils.module import look_up_option


class BaseDetectionNetwork(nn.Module):
    """
    Base detection network class.

    Args:
        spatial_dims: number of spatial dimensions of the images. We support both 2D and 3D images.
        num_classes: number of output classes of the model (including the background).
        size_divisible: the spatial size of the network input should be divisible by size_divisible.
    """

    def __init__(
        self, spatial_dims: int, num_classes: int, size_divisible: Union[Sequence[int], int] = 1, *args, **kwargs
    ) -> None:
        super().__init__()
        self.spatial_dims = look_up_option(spatial_dims, supported=[1, 2, 3])
        self.num_classes = num_classes
        self.size_divisible = ensure_tuple_rep(size_divisible, self.spatial_dims)

    @abstractmethod
    def forward(self, images: Tensor):
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")

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

from .transforms_tensor import Flip
from .compose import MapTransform


class Flipd(MapTransform):
    """Dictionary-based wrapper of :py:class:`monai.transforms.transfroms_tensor.Flip`.

    Args:
        keys (dict): Keys to pick data for transformation.
        spatial_axis (tuple or list of ints): spatial axes along which to flip over.
    """

    def __init__(self, keys, spatial_axis):
        super().__init__(keys)
        self.flipper = Flip(spatial_axis=spatial_axis)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = self.flipper(d[key])
        return d

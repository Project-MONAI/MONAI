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


from monai.data.meta_tensor import MetaTensor
from monai.transforms.inverse import InvertibleTransform


class ApplyPendingd(InvertibleTransform):
    """
    Apply wraps the apply method and can function as a Transform in either array or dictionary
    mode.
    """

    def __init__(self, keys):
        super().__init__()
        self.keys = keys

    def __call__(self, data, **kwargs):
        if not isinstance(data, dict):
            raise ValueError("'data' must be of type dict but is '{type(data)}'")

        rd = dict(data)
        for k in self.keys:
            if isinstance(rd[k], MetaTensor):
                rd[k] = apply_pending(rd[k], **kwargs)
        return rd

    def inverse(self, data):
        if not isinstance(data, dict):
            raise ValueError("'data' must be of type dict but is '{type(data)}'")

        return self(data)

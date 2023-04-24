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


from __future__ import annotations

from monai.data.meta_tensor import MetaTensor
from monai.transforms.inverse import InvertibleTransform
from monai.transforms.lazy.functional import apply_pending


class ApplyPending(InvertibleTransform):
    """
    Apply wraps the apply_pending method and can function as a Transform in an array-based pipeline.
    """

    def __init__(self):
        super().__init__()

    def __call__(self, data, *args, **kwargs):
        if isinstance(data, MetaTensor):
            return apply_pending(data, *args, **kwargs)

        return data

    def inverse(self, data):
        return self(data)

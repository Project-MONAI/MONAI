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

from monai.transforms.lazy.functional import apply
from monai.transforms.inverse import InvertibleTransform

__all__ = ["Apply"]


class Apply(InvertibleTransform):
    """
    Apply wraps the apply method and can function as a Transform in either array or dictionary
    mode.
    """
    def __init__(self):
        super().__init__()

    def __call__(self, *args, **kwargs):
        return apply(*args, **kwargs)

    def inverse(self, data):
        return NotImplementedError()


# class Applyd(MapTransform, InvertibleTransform):
#
#     def __init__(self):
#         super().__init__()
#
#     def __call__(
#             self,
#             d: dict
#     ):
#         rd = dict()
#         for k, v in d.items():
#             rd[k] = apply(v)
#
#     def inverse(self, data):
#         return NotImplementedError()

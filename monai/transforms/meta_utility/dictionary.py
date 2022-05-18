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
A collection of dictionary-based wrappers for moving between MetaTensor types and dictionaries of data.
These can be used to make backwards compatible code.

Class names are ended with 'd' to denote dictionary-based transforms.
"""

from copy import deepcopy
from typing import Dict, Hashable, Mapping

from monai.config.type_definitions import NdarrayOrTensor
from monai.data.meta_tensor import MetaTensor
from monai.transforms.inverse import InvertibleTransform
from monai.transforms.transform import MapTransform
from monai.utils.enums import PostFix, TransformBackends

__all__ = [
    "FromMetaTensord",
    "FromMetaTensorD",
    "FromMetaTensorDict",
    "ToMetaTensord",
    "ToMetaTensorD",
    "ToMetaTensorDict",
]


class FromMetaTensord(MapTransform, InvertibleTransform):
    """
    Dictionary-based transform to convert MetaTensor to a dictionary.

    If input is `{"a": MetaTensor, "b": MetaTensor}`, then output will
    have the form `{"a": torch.Tensor, "a_meta_dict": dict, "a_transforms": list, "b": ...}`.
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            im: MetaTensor = d[key]  # type: ignore
            d.update(im.as_dict(key))
            self.push_transform(d, key)
        return d

    def inverse(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = deepcopy(dict(data))
        for key in self.key_iterator(d):
            # check transform
            _ = self.get_most_recent_transform(d, key)
            # do the inverse
            im = d[key]
            meta = d.pop(PostFix.meta(key), None)
            transforms = d.pop(PostFix.transforms(key), None)
            im = MetaTensor(im, meta=meta, applied_operations=transforms)  # type: ignore
            d[key] = im
            # Remove the applied transform
            self.pop_transform(d, key)
        return d


class ToMetaTensord(MapTransform, InvertibleTransform):
    """
    Dictionary-based transform to convert a dictionary to MetaTensor.

    If input is `{"a": torch.Tensor, "a_meta_dict": dict, "b": ...}`, then output will
    have the form `{"a": MetaTensor, "b": MetaTensor}`.
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            self.push_transform(d, key)
            im = d[key]
            meta = d.pop(PostFix.meta(key), None)
            transforms = d.pop(PostFix.transforms(key), None)
            im = MetaTensor(im, meta=meta, applied_operations=transforms)  # type: ignore
            d[key] = im
        return d

    def inverse(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = deepcopy(dict(data))
        for key in self.key_iterator(d):
            # check transform
            _ = self.get_most_recent_transform(d, key)
            # do the inverse
            im: MetaTensor = d[key]  # type: ignore
            d.update(im.as_dict(key))
            # Remove the applied transform
            self.pop_transform(d, key)
        return d


FromMetaTensorD = FromMetaTensorDict = FromMetaTensord
ToMetaTensorD = ToMetaTensorDict = ToMetaTensord

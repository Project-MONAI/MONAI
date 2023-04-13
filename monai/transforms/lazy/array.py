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
from typing import Callable

import torch
from monai.data.meta_tensor import MetaTensor

from monai.transforms.lazy.functional import apply_pending
from monai.transforms.inverse import InvertibleTransform

__all__ = ["ApplyPending"]


class ApplyPending(InvertibleTransform):
    """
    Apply wraps the apply method and can function as a Transform in either array or dictionary
    mode.
    """

    def __init__(self):
        super().__init__()

    def __call__(self, data, *args, **kwargs):
        if isinstance(data, dict):
            rd = dict(data)
            for k, v in data.items():
                if isinstance(v, MetaTensor):
                    rd[k] = apply_pending(v, *args, **kwargs)
            return rd

        return apply_pending(data, *args, **kwargs)

    def inverse(self, data):
        return self(data)


class CacheMechanism:
    """
    The interface for caching mechanisms to be used with CachedTransform. This interface provides
    the ability to check whether cached objects are present, test and fetch simultaneously, and
    store items. It makes no other assumptions about the caching mechanism, capacity, cache eviction
    strategies or any other aspect of cache implementation
    """

    def try_fetch(
            self,
            key
    ):
        raise NotImplementedError()

    def store(
            self,
            key,
            value
    ):
        raise NotImplementedError()


class CachedTransform:
    """
    CachedTransformCompose provides the functionality to cache the output of one or more transforms
    such that they only need to be run once. Each time that CachedTransform is run, it checks whether
    a cached entity is present, and if that entity is present, it loads it and returns the
    resulting tensor / tensors as output. If that entity is not present in the cache, it executes
    the transforms in its internal pipeline and caches the result before returning it.
    """

    def __init__(
            self,
            transforms: Callable,
            cache: CacheMechanism
    ):
        """
        Args:
        transforms: A sequence of callable objects
        cache: A caching mechanism that implements the `CacheMechanism` interface
        """
        self.transforms = transforms
        self.cache = cache

    def __call__(
            self,
            key,
            *args,
            **kwargs
    ):
        is_present, value = self.cache.try_fetch(key)

        if is_present:
            return value

        result = self.transforms(*args, **kwargs)
        self.cache.store(key, result)

        return result


class MultiSampleTransform:
    """
    MultiSampleTransformCompose takes the output of a transform that generates multiple samples
    and executes each sample separately in a depth first fashion, gathering the results into an
    array that is finally returned after all samples are processed
    """
    def __init__(
            self,
            multi_sample: Callable,
            transforms: Callable,
    ):
        self.multi_sample = multi_sample
        self.transforms = transforms

    def __call__(
            self,
            t,
            *args,
            **kwargs
    ):
        output = list()
        for mt in self.multi_sample(t):
            mt_out = self.transforms(mt)
            if isinstance(mt_out, (torch.Tensor, dict)):
                output.append(mt_out)
            elif isinstance(mt_out, list):
                output += mt_out

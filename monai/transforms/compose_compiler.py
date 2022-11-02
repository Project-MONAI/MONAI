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

from monai.transforms import LazyTrait, MultiSampleTrait, RandomizableTrait
from monai.transforms.lazy.array import Apply
from monai.transforms.utility.array import CachedTransform, MultiSampleTransform


class ComposeCompiler:

    def compile(self, transforms, cache_mechanism):

        transforms1 = self.compile_caching(transforms, cache_mechanism)

        transforms2 = self.compile_multisampling(transforms1)

        transforms3 = self.compile_lazy_resampling(transforms2)

        return transforms3

    def compile_caching(self, transforms, cache_mechanism):
        # TODO: handle being passed a transform list with containers
        # given a list of transforms, determine where to add a cached transform object
        # and what transforms to put in it
        cacheable = list()
        for t in transforms:
            if self.transform_is_random(t) is False:
                cacheable.append(t)
            else:
                break

        if len(cacheable) == 0:
            return list(transforms)
        else:
            return [CachedTransform(cacheable, cache_mechanism)] + transforms[len(cacheable):]

    def compile_multisampling(self, transforms):
        for i in reversed(range(len(transforms))):
            if self.transform_is_multisampling(transforms[i]) is True:
                transforms_ = transforms[:i] + [MultiSampleTransform(transforms[i],
                                                                            transforms[i+1:])]
                return self.compile_multisampling(transforms_)

        return list(transforms)

    def compile_lazy_resampling(self, transforms):
        result = list()
        lazy = list()
        for i in range(len(transforms)):
            if self.transform_is_lazy(transforms[i]):
                lazy.append(transforms[i])
            else:
                if len(lazy) > 0:
                    result.extend(lazy)
                    result.append(Apply())
                    lazy = list()
                result.append(transforms[i])
        if len(lazy) > 0:
            result.extend(lazy)
            result.append(Apply())
        return result

    @staticmethod
    def transform_is_random(t):
        return isinstance(t, RandomizableTrait)

    @staticmethod
    def transform_is_container(t):
        return isinstance(t, (CachedTransform, MultiSampleTransform))

    @staticmethod
    def transform_is_multisampling(t):
        return isinstance(t, MultiSampleTrait)

    @staticmethod
    def transform_is_lazy(t):
        return isinstance(t, LazyTrait)

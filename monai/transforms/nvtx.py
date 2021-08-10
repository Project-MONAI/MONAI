# Copyright 2020 - 2021 MONAI Consortium
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
Wrapper around NVIDIA Tools Extension for profiling MONAI transformations
"""

from monai.transforms.transform import RandomizableTransform, Transform
from monai.utils import optional_import

_nvtx, _ = optional_import("torch._C._nvtx", descriptor="NVTX is not installed. Are you sure you have a CUDA build?")

__all__ = ["RangePush", "RandRangePush", "RangePop", "RandRangePop", "Mark", "RandMark"]


class RangePush(Transform):
    """
    Pushes a range onto a stack of nested range span.
    Stores zero-based depth of the range that is started.

    Args:
        msg: ASCII message to associate with range
    """

    def __init__(self, msg: str) -> None:
        self.msg = msg
        self.depth = None

    def __call__(self, data):
        self.depth = _nvtx.rangePushA(self.msg)
        return data


class RandRangePush(RandomizableTransform):
    """
    Pushes a range onto a stack of nested range span (RandomizableTransform).
    Stores zero-based depth of the range that is started.

    Args:
        msg: ASCII message to associate with range
    """

    def __init__(self, msg: str) -> None:
        super().__init__()
        self.msg = msg
        self.depth = None

    def __call__(self, data):
        self.depth = _nvtx.rangePushA(self.msg)
        return data


class RangePop(Transform):
    """
    Pops a range off of a stack of nested range spans.
    Stores zero-based depth of the range that is ended.
    """

    def __call__(self, data):
        _nvtx.rangePop()
        return data


class RandRangePop(RandomizableTransform):
    """
    Pops a range off of a stack of nested range spans (RandomizableTransform).
    Stores zero-based depth of the range that is ended.
    """

    def __init__(self):
        super().__init__()

    def __call__(self, data):
        _nvtx.rangePop()
        return data


class Mark(Transform):
    """
    Mark an instantaneous event that occurred at some point.

    Args:
        msg: ASCII message to associate with the event.
    """

    def __init__(self, msg: str) -> None:
        self.msg = msg

    def __call__(self, data):
        _nvtx.markA(self.msg)
        return data


class RandMark(RandomizableTransform):
    """
    Mark an instantaneous event that occurred at some point.
    (RandomizableTransform)

    Args:
        msg: ASCII message to associate with the event.
    """

    def __init__(self, msg: str) -> None:
        super().__init__()
        self.msg = msg

    def __call__(self, data):
        _nvtx.markA(self.msg)
        return data


MarkDict = MarkD = Markd = Mark
RandMarkDict = RandMarkD = RandMarkd = RandMark
RandRangePopDict = RandRangePopD = RandRangePopd = RandRangePop
RandRangePushDict = RandRangePushD = RandRangePushd = RandRangePush
RangePopDict = RangePopD = RangePopd = RangePop
RangePushDict = RangePushD = RangePushd = RangePush

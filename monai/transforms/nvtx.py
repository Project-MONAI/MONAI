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

from typing import Optional

from monai.transforms.transform import RandomizableTransform, Transform
from monai.utils import optional_import

_nvtx, _ = optional_import("torch._C._nvtx", descriptor="NVTX is not installed. Are you sure you have a CUDA build?")

__all__ = [
    "Mark",
    "Markd",
    "MarkD",
    "MarkDict",
    "RandMark",
    "RandMarkd",
    "RandMarkD",
    "RandMarkDict",
    "RandRange",
    "RandRanged",
    "RandRangeD",
    "RandRangeDict",
    "RandRangePop",
    "RandRangePopd",
    "RandRangePopD",
    "RandRangePopDict",
    "RandRangePush",
    "RandRangePushd",
    "RandRangePushD",
    "RandRangePushDict",
    "Range",
    "Ranged",
    "RangeD",
    "RangeDict",
    "RangePop",
    "RangePopd",
    "RangePopD",
    "RangePopDict",
    "RangePush",
    "RangePushd",
    "RangePushD",
    "RangePushDict",
]


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


class RandRangePush(RangePush, RandomizableTransform):
    """
    Pushes a range onto a stack of nested range span (RandomizableTransform).
    Stores zero-based depth of the range that is started.

    Args:
        msg: ASCII message to associate with range
    """


class RangePop(Transform):
    """
    Pops a range off of a stack of nested range spans.
    Stores zero-based depth of the range that is ended.
    """

    def __call__(self, data):
        _nvtx.rangePop()
        return data


class RandRangePop(RangePop, RandomizableTransform):
    """
    Pops a range off of a stack of nested range spans (RandomizableTransform).
    Stores zero-based depth of the range that is ended.
    """


class Range(Transform):
    """
    Pushes an NVTX range before a transform, and pops it afterwards.
    Stores zero-based depth of the range that is started.

    Args:
        msg: ASCII message to associate with range
    """

    def __init__(self, transform: Transform, msg: Optional[str] = None) -> None:
        if msg is None:
            msg = type(transform).__name__
        self.msg = msg
        self.transform = transform
        self.depth = None

    def __call__(self, data):
        self.depth = _nvtx.rangePushA(self.msg)
        data = self.transform(data)
        _nvtx.rangePop()
        return data


class RandRange(Range, RandomizableTransform):
    """
    Pushes an NVTX range at the before a transfrom, and pops it afterwards.(RandomizableTransform).
    Stores zero-based depth of the range that is ended.
    """


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


class RandMark(Mark, RandomizableTransform):
    """
    Mark an instantaneous event that occurred at some point.
    (RandomizableTransform)

    Args:
        msg: ASCII message to associate with the event.
    """


RangePushDict = RangePushD = RangePushd = RangePush
RandRangePushDict = RandRangePushD = RandRangePushd = RandRangePush

RangePopDict = RangePopD = RangePopd = RangePop
RandRangePopDict = RandRangePopD = RandRangePopd = RandRangePop

RangeDict = RangeD = Ranged = Range
RandRangeDict = RandRangeD = RandRanged = RandRange

MarkDict = MarkD = Markd = Mark
RandMarkDict = RandMarkD = RandMarkd = RandMark

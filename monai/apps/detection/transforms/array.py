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
A collection of "vanilla" transforms for box operations
https://github.com/Project-MONAI/MONAI/wiki/MONAI_Design
"""

from typing import Type, Union

from monai.config.type_definitions import NdarrayOrTensor
from monai.data.box_utils import BoxMode, convert_box_mode, convert_box_to_standard_mode
from monai.transforms.transform import Transform
from monai.utils.enums import TransformBackends

__all__ = ["ConvertBoxToStandardMode", "ConvertBoxMode"]


class ConvertBoxMode(Transform):
    """
    This transform converts the boxes in src_mode to the dst_mode.

    Example:
        .. code-block:: python

            boxes = torch.ones(10,4)
            # convert boxes with format [xmin, ymin, xmax, ymax] to [xcenter, ycenter, xsize, ysize].
            box_converter = ConvertBoxMode(src_mode="xyxy", dst_mode="ccwh")
            box_converter(boxes)
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(
        self,
        src_mode: Union[str, BoxMode, Type[BoxMode], None] = None,
        dst_mode: Union[str, BoxMode, Type[BoxMode], None] = None,
    ) -> None:
        """

        Args:
            src_mode: source box mode. If it is not given, this func will assume it is ``StandardMode()``.
            dst_mode: target box mode. If it is not given, this func will assume it is ``StandardMode()``.

        Note:
            ``StandardMode`` = :class:`~monai.data.box_utils.CornerCornerModeTypeA`,
            also represented as "xyxy" for 2D and "xyzxyz" for 3D.

            src_mode and dst_mode can be:
                #. str: choose from :class:`~monai.utils.enums.BoxModeName`, for example,
                    - "xyxy": boxes has format [xmin, ymin, xmax, ymax]
                    - "xyzxyz": boxes has format [xmin, ymin, zmin, xmax, ymax, zmax]
                    - "xxyy": boxes has format [xmin, xmax, ymin, ymax]
                    - "xxyyzz": boxes has format [xmin, xmax, ymin, ymax, zmin, zmax]
                    - "xyxyzz": boxes has format [xmin, ymin, xmax, ymax, zmin, zmax]
                    - "xywh": boxes has format [xmin, ymin, xsize, ysize]
                    - "xyzwhd": boxes has format [xmin, ymin, zmin, xsize, ysize, zsize]
                    - "ccwh": boxes has format [xcenter, ycenter, xsize, ysize]
                    - "cccwhd": boxes has format [xcenter, ycenter, zcenter, xsize, ysize, zsize]
                #. BoxMode class: choose from the subclasses of :class:`~monai.data.box_utils.BoxMode`, for example,
                    - CornerCornerModeTypeA: equivalent to "xyxy" or "xyzxyz"
                    - CornerCornerModeTypeB: equivalent to "xxyy" or "xxyyzz"
                    - CornerCornerModeTypeC: equivalent to "xyxy" or "xyxyzz"
                    - CornerSizeMode: equivalent to "xywh" or "xyzwhd"
                    - CenterSizeMode: equivalent to "ccwh" or "cccwhd"
                #. BoxMode object: choose from the subclasses of :class:`~monai.data.box_utils.BoxMode`, for example,
                    - CornerCornerModeTypeA(): equivalent to "xyxy" or "xyzxyz"
                    - CornerCornerModeTypeB(): equivalent to "xxyy" or "xxyyzz"
                    - CornerCornerModeTypeC(): equivalent to "xyxy" or "xyxyzz"
                    - CornerSizeMode(): equivalent to "xywh" or "xyzwhd"
                    - CenterSizeMode(): equivalent to "ccwh" or "cccwhd"
                #. None: will assume mode is ``StandardMode()``
        """
        self.src_mode = src_mode
        self.dst_mode = dst_mode

    def __call__(self, boxes: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Converts the boxes in src_mode to the dst_mode.

        Returns:
            bounding boxes with target mode, with same data type as ``boxes``, does not share memory with ``boxes``
        """
        return convert_box_mode(boxes, src_mode=self.src_mode, dst_mode=self.dst_mode)


class ConvertBoxToStandardMode(Transform):
    """
    Convert given boxes to standard mode.
    Standard mode is "xyxy" or "xyzxyz",
    representing box format of [xmin, ymin, xmax, ymax] or [xmin, ymin, zmin, xmax, ymax, zmax].

    Example:
        .. code-block:: python

            boxes = torch.ones(10,6)
            # convert boxes with format [xmin, xmax, ymin, ymax, zmin, zmax] to [xmin, ymin, zmin, xmax, ymax, zmax]
            box_converter = ConvertBoxToStandardMode(mode="xxyyzz")
            box_converter(boxes)
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, mode: Union[str, BoxMode, Type[BoxMode], None] = None) -> None:
        """
        Args:
            mode: source box mode. If it is not given, this func will assume it is ``StandardMode()``.
                It follows the same format with ``src_mode`` in :class:`~monai.apps.detection.transforms.array.ConvertBoxMode` .
        """
        self.mode = mode

    def __call__(self, boxes: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Convert given boxes to standard mode.
        Standard mode is "xyxy" or "xyzxyz",
        representing box format of [xmin, ymin, xmax, ymax] or [xmin, ymin, zmin, xmax, ymax, zmax].

        Returns:
            bounding boxes with standard mode, with same data type as ``boxes``, does not share memory with ``boxes``
        """
        return convert_box_to_standard_mode(boxes, mode=self.mode)

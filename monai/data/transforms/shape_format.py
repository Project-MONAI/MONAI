
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


import numpy as np


class ShapeFormat:
    """ShapeFormat defines meanings for the data in a MedicalImage.
    Image data is a numpy's ndarray. Without shape format, it is impossible to know what each
    dimension means.

    NOTE: ShapeFormat objects are immutable.

    """

    CHW = 'CHW'
    CHWD = 'CHWD'


def get_shape_format(img: np.ndarray):
    """Return the shape format of the image data

    Args:
        img (np.ndarray): the image data

    Returns: a shape format or None

    Raise: AssertionError if any of the specified args is invalid

    """
    assert isinstance(img, np.ndarray), 'invalid value img - must be np.ndarray'
    if img.ndim == 3:
        return ShapeFormat.CHW
    elif img.ndim == 4:
        return ShapeFormat.CHWD
    else:
        return None

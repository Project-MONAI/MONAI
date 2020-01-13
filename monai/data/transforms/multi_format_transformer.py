
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


import logging
import numpy as np
from .shape_format import ShapeFormat
from .shape_format import get_shape_format


class MultiFormatTransformer:
    """Base class for multi-format transformer.

    12 numpy data formats are specified based on image dimension, batch mode, and channel mode
    """

    def __init__(self):

        self._format_handlers = {
            ShapeFormat.CHWD: self._handle_chwd,
            ShapeFormat.CHW: self._handle_chw
        }
        self._logger = logging.getLogger(self.__class__.__name__)

    def _handle_any(self, *args, **kwargs):
        return None

    def _handle_chw(self, *args, **kwargs):
        return None

    def _handle_chwd(self, *args, **kwargs):
        return None

    def transform(self, img, *args, **kwargs):

        assert isinstance(img, np.ndarray), 'img must be np.ndarray'

        shape_format = get_shape_format(img)
        if not shape_format:
            raise ValueError('the image data has invalid shape format')

        h = self._format_handlers.get(shape_format, None)
        if h is None:
            raise ValueError('unsupported image shape format: {}'.format(shape_format))

        result = h(img, *args, **kwargs)
        if result is not None:
            return result

        result = self._handle_any(img, *args, **kwargs)

        if result is None:
            raise NotImplementedError(
                'transform {} does not support format {}'.format(self.__class__.__name__, shape_format))

        return result

    def __call__(self, *args, **kwargs):
        return self.transform(*args, **kwargs)

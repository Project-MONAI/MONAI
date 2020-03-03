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

class ActivationFunc:
    """Commonly used activation function names.
    """

    SOFTMAX = "softmax"
    LOG_SOFTMAX = "log_softmax"
    SIGMOID = "sigmoid"
    LINEAR = "linear"
    TANH = "tanh"


class DataElementKey:
    """Data Element keys
    """

    IMAGE = "image"
    LABEL = "label"


class ImageProperty:
    """Key names for image properties.
    """

    FILENAME_OR_OBJ = 'filename_or_obj'
    AFFINE = 'affine'  # image affine matrix
    ORIGINAL_AFFINE = 'original_affine'  # original affine matrix before transformation
    SPACING = 'spacing'  # itk naming convention for pixel/voxel size
    AS_CLOSEST_CANONICAL = 'as_closest_canonical'  # load the image as closest to canonical axis format
    BACKGROUND_INDEX = 'background_index'  # which index is background


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


class ImageProperty:
    """Key names for image properties.

    """
    DATA = 'data'
    FILENAME = 'file_name'
    AFFINE = 'affine'  # image affine matrix
    ORIGINAL_SHAPE = 'original_shape'
    ORIGINAL_SHAPE_FORMAT = 'original_shape_format'
    SPACING = 'spacing'  # itk naming convention for pixel/voxel size
    FORMAT = 'file_format'
    NIFTI_FORMAT = 'nii'
    IS_CANONICAL = 'is_canonical'
    SHAPE_FORMAT = 'shape_format'
    BACKGROUND_INDEX = 'background_index'  # which index is background

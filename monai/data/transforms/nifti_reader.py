
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


import nibabel as nib
import numpy as np

from .image_props import ImageProperty
from .image_reader import ImageReader


class NiftiReader(ImageReader):
    """ Reads nifti files.

    Args:
        dtype(np) : type for loaded data.
        nii_is_channels(bool): Is nifti channels first. (Default: False)
        as_closest_canonical (bool): Load in canonical orientation. (Default: True)

    Returns:
        img: image data
        img_props: dict of image properties

    """

    def __init__(self, dtype=np.float32, nii_is_channels_first=False, as_closest_canonical=True):
        ImageReader.__init__(self, dtype)

        # Make a list of fields to be loaded
        self.nii_is_channels_first = nii_is_channels_first
        self.as_closest_canonical = as_closest_canonical
        self._dtype = dtype

    def _load_data(self, file_name):
        self._logger.debug("Loading nifti file {}".format(file_name))
        epi_img = nib.load(file_name)
        assert epi_img is not None

        if self.as_closest_canonical:
            epi_img = nib.as_closest_canonical(epi_img)

        img_array = epi_img.get_fdata(dtype=self._dtype)

        affine = epi_img.affine
        shape = epi_img.header.get_data_shape()
        spacing = epi_img.header.get_zooms()
        if len(spacing) > 3:  # Possible temporal spacing in 4th dimension
            spacing = spacing[:3]
        return img_array, affine, shape, spacing, self.as_closest_canonical

    def _read_from_file(self, file_name):
        """ Loads a nifti file.

        Args:
            file_name (str): path to nifti file.

        Returns:
            Loaded MedicalImage.
        """
        img_array, affine, shape, spacing, is_canonical = self._load_data(file_name)
        num_dims = len(img_array.shape)
        img_array = img_array.astype(self._dtype)

        if num_dims == 2:
            img_array = np.expand_dims(img_array, axis=0)
        elif num_dims == 3:
            img_array = np.expand_dims(img_array, axis=0)
        elif num_dims <= 5:
            # if 4d data, we assume 4th dimension is channels.
            # if 5d data, try to squeeze 5th dimension.
            if num_dims == 5:
                img_array = np.squeeze(img_array)
                if len(img_array.shape) != 4:
                    raise ValueError("NiftiReader doesn't support time based data.")

            if not self.nii_is_channels_first:
                # convert to channel first
                img_array = np.transpose(img_array, (3, 0, 1, 2))
        else:
            raise NotImplementedError('NifitReader does not support image of dims {}'.format(num_dims))

        img_props = {
            ImageProperty.AFFINE: affine,
            ImageProperty.FILENAME: file_name,
            ImageProperty.FORMAT: ImageProperty.NIFTI_FORMAT,
            ImageProperty.ORIGINAL_SHAPE: shape,
            ImageProperty.SPACING: spacing,
            ImageProperty.IS_CANONICAL: is_canonical
        }

        return img_array, img_props

    def _read_from_file_list(self, file_names):
        """Loads a multi-channel nifti file (1 channel per file)

        Args:
            file_names (list): list of file names.

        Returns:
            Loaded MedicalImage.
        """
        img_array = []
        affine = None
        shape = None
        spacing = None
        is_canonical = None

        for file_name in file_names:
            _img_array, _affine, _shape, _spacing, _is_canonical = self._load_data(file_name)

            # Check if next data array matches the previous one
            # warnings if affine or spacing does not match
            if affine is None:
                affine = _affine
            elif not np.array_equal(_affine, affine):
                self._logger.warning(
                    'Affine matrix of [{}] is not consistent with previous data entry'.format(file_name))

            if spacing is None:
                spacing = _spacing
            elif _spacing != spacing:
                self._logger.warning(
                    'Spacing of [{}] is not consistent with previous data entry'.format(file_name))

            # error if shapes do not match as this will cause errors later
            if shape is None:
                shape = _shape
            elif _shape != shape:
                error_message = 'Shape of [{}] is not consistent with previous data entry' \
                    .format(file_name)

                self._logger.error(error_message)
                raise ValueError(error_message)

            # Check if canonical settings are same.
            if is_canonical is None:
                is_canonical = _is_canonical
            elif _is_canonical != is_canonical:
                self._logger.warning(
                    'File {} is loaded in different canonical settings than previous files.'.format(file_name))

            # append image array for stacking
            img_array.append(_img_array)

        # load and stack channels along first dimension
        img_array = np.stack(img_array, axis=0)
        shape = np.shape(img_array)  # update to new shape
        num_dims = len(shape)
        img_array = img_array.astype(self._dtype)

        if num_dims != 3 and num_dims != 4:
            raise NotImplementedError('NiftiReader does not support image of dims {}'.format(num_dims))

        img_props = {
            ImageProperty.AFFINE: affine,
            ImageProperty.FILENAME: file_names,
            ImageProperty.FORMAT: ImageProperty.NIFTI_FORMAT,
            ImageProperty.ORIGINAL_SHAPE: shape,
            ImageProperty.SPACING: spacing,
            ImageProperty.IS_CANONICAL: is_canonical
        }

        return img_array, img_props

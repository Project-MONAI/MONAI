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
import nibabel as nib
import random

from torch.utils.data import Dataset

from monai.utils.moduleutils import export


def load_nifti(filename_or_obj, as_closest_canonical=False, image_only=True, dtype=None):
    """
    Loads a Nifti file from the given path or file-like object.

    Args:
        filename_or_obj (str or file): path to file or file-like object
        as_closest_canonical (bool): if True, load the image as closest to canonical axis format
        image_only (bool): if True return only the image volume, other return image volume and header dict
        dtype (np.dtype, optional): if not None convert the loaded image to this data type

    Returns:
        The loaded image volume if `image_only` is True, or a tuple containing the volume and the Nifti
        header in dict format otherwise
    """

    img = nib.load(filename_or_obj)

    if as_closest_canonical:
        img = nib.as_closest_canonical(img)

    if dtype is not None:
        dat = img.get_fdata(dtype=dtype)
    else:
        dat = np.asanyarray(img.dataobj)

    header = dict(img.header)
    header['filename_or_obj'] = filename_or_obj

    if image_only:
        return dat
    else:
        return dat, header


@export("monai.data.readers")
class NiftiDataset(Dataset):
    """
    Loads image/segmentation pairs of Nifti files from the given filename lists. Transformations can be specified
    for the image and segmentation arrays separately.
    """

    def __init__(self, image_files, seg_files, transform=None, seg_transform=None):
        """
        Initializes the dataset with the image and segmentation filename lists. The transform `transform` is applied
        to the images and `seg_transform` to the segmentations.

        Args:
            image_files (list of str): list of image filenames
            seg_files (list of str): list of segmentation filenames
            transform (Callable, optional): transform to apply to image arrays
            seg_transform (Callable, optional): transform to apply to segmentation arrays
        """

        if len(image_files) != len(seg_files):
            raise ValueError('Must have same number of image and segmentation files')

        self.image_files = image_files
        self.seg_files = seg_files
        self.transform = transform 
        self.seg_transform = seg_transform 

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        img = load_nifti(self.image_files[index])
        seg = load_nifti(self.seg_files[index])

        # https://github.com/pytorch/vision/issues/9#issuecomment-304224800
        seed = np.random.randint(2147483647)

        if self.transform is not None:
            random.seed(seed)
            img = self.transform(img)

        if self.seg_transform is not None:
            random.seed(seed)  # ensure randomized transforms roll the same values for segmentations as images
            seg = self.seg_transform(seg)

        return img, seg

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
from torch.utils.data._utils.collate import np_str_obj_array_pattern

from monai.utils.module import export


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

    Note:
        header['original_affine'] stores the original affine loaded from `filename_or_obj`.
        header['affine'] stores the affine after the optional `as_closest_canonical` transform.
    """

    img = nib.load(filename_or_obj)

    header = dict(img.header)
    header['filename_or_obj'] = filename_or_obj
    header['original_affine'] = img.affine
    header['affine'] = img.affine
    header['as_closest_canonical'] = as_closest_canonical

    if as_closest_canonical:
        img = nib.as_closest_canonical(img)
        header['affine'] = img.affine

    if dtype is not None:
        dat = img.get_fdata(dtype=dtype)
    else:
        dat = np.asanyarray(img.dataobj)

    if image_only:
        return dat
    return dat, header


@export("monai.data")
class NiftiDataset(Dataset):
    """
    Loads image/segmentation pairs of Nifti files from the given filename lists. Transformations can be specified
    for the image and segmentation arrays separately.
    """

    def __init__(self, image_files, seg_files, as_closest_canonical=False,
                 transform=None, seg_transform=None, image_only=True, dtype=None):
        """
        Initializes the dataset with the image and segmentation filename lists. The transform `transform` is applied
        to the images and `seg_transform` to the segmentations.

        Args:
            image_files (list of str): list of image filenames
            seg_files (list of str): list of segmentation filenames
            as_closest_canonical (bool): if True, load the image as closest to canonical orientation
            transform (Callable, optional): transform to apply to image arrays
            seg_transform (Callable, optional): transform to apply to segmentation arrays
            image_only (bool): if True return only the image volume, other return image volume and header dict
            dtype (np.dtype, optional): if not None convert the loaded image to this data type
        """

        if len(image_files) != len(seg_files):
            raise ValueError('Must have same number of image and segmentation files')

        self.image_files = image_files
        self.seg_files = seg_files
        self.as_closest_canonical = as_closest_canonical
        self.transform = transform
        self.seg_transform = seg_transform
        self.image_only = image_only
        self.dtype = dtype

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        meta_data = None
        if self.image_only:
            img = load_nifti(self.image_files[index], as_closest_canonical=self.as_closest_canonical,
                             image_only=self.image_only, dtype=self.dtype)
        else:
            img, meta_data = load_nifti(self.image_files[index], as_closest_canonical=self.as_closest_canonical,
                                        image_only=self.image_only, dtype=self.dtype)
        seg = load_nifti(self.seg_files[index])

        # https://github.com/pytorch/vision/issues/9#issuecomment-304224800
        seed = np.random.randint(2147483647)

        if self.transform is not None:
            random.seed(seed)
            img = self.transform(img)

        if self.seg_transform is not None:
            random.seed(seed)  # ensure randomized transforms roll the same values for segmentations as images
            seg = self.seg_transform(seg)

        if self.image_only or meta_data is None:
            return img, seg

        compatible_meta = {}
        for meta_key in meta_data:
            meta_datum = meta_data[meta_key]
            if type(meta_datum).__name__ == 'ndarray' \
                    and np_str_obj_array_pattern.search(meta_datum.dtype.str) is not None:
                continue
            compatible_meta[meta_key] = meta_datum
        return img, seg, compatible_meta

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
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import np_str_obj_array_pattern

from monai.data.utils import correct_nifti_header_if_necessary
from monai.transforms.compose import Randomizable
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
    img = correct_nifti_header_if_necessary(img)

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

    def __init__(self, image_files, seg_files=None, labels=None, as_closest_canonical=False,
                 transform=None, seg_transform=None, image_only=True, dtype=None):
        """
        Initializes the dataset with the image and segmentation filename lists. The transform `transform` is applied
        to the images and `seg_transform` to the segmentations.

        Args:
            image_files (list of str): list of image filenames
            seg_files (list of str): if in segmentation task, list of segmentation filenames
            labels (list or array): if in classification task, list of classification labels
            as_closest_canonical (bool): if True, load the image as closest to canonical orientation
            transform (Callable, optional): transform to apply to image arrays
            seg_transform (Callable, optional): transform to apply to segmentation arrays
            image_only (bool): if True return only the image volume, other return image volume and header dict
            dtype (np.dtype, optional): if not None convert the loaded image to this data type
        """

        if seg_files is not None and len(image_files) != len(seg_files):
            raise ValueError('Must have same number of image and segmentation files')

        self.image_files = image_files
        self.seg_files = seg_files
        self.labels = labels
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
        target = None
        if self.seg_files is not None:
            target = load_nifti(self.seg_files[index])
        elif self.labels is not None:
            target = self.labels[index]

        seed = np.random.randint(2147483647)

        if self.transform is not None:
            if isinstance(self.transform, Randomizable):
                self.transform.set_random_state(seed=seed)
            img = self.transform(img)

        if self.seg_transform is not None:
            if isinstance(self.seg_transform, Randomizable):
                self.seg_transform.set_random_state(seed=seed)
            target = self.seg_transform(target)

        if self.image_only or meta_data is None:
            return img, target

        compatible_meta = {}
        for meta_key in meta_data:
            meta_datum = meta_data[meta_key]
            if type(meta_datum).__name__ == 'ndarray' \
                    and np_str_obj_array_pattern.search(meta_datum.dtype.str) is not None:
                continue
            compatible_meta[meta_key] = meta_datum
        return img, target, compatible_meta

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

import itertools
from typing import Dict, Sequence

import numpy as np
from joblib import Parallel, delayed

from monai.transforms import LoadImaged


class DatasetCalculator:
    """
    This class provides a way to calculate a reasonable output voxel spacing according to
    the input dataset. The achieved values can used to resample the input in 3d segmentation tasks
    (like using as the `pixel` parameter in `monai.transforms.Spacingd`).
    In addition, it also supports to count the mean, std, min and max intensities of the input,
    and these statistics are helpful for image normalization
    (like using in `monai.transforms.ScaleIntensityRanged` and `monai.transforms.NormalizeIntensityd`).

    The algorithm for calculation refers to:
    `Automated Design of Deep Learning Methods for Biomedical Image Segmentation <https://arxiv.org/abs/1904.08128>`_.

    """

    def __init__(
        self,
        datalist: Sequence[Dict],
        image_key: str = "image",
        label_key: str = "label",
        meta_key_postfix: str = "meta_dict",
        num_workers: int = -1,
    ):
        """
        Args:
            datalist: a list that contains the path of all images and labels. The list is
                consisted with dictionaries, and each dictionary contains the image and label
                path of one sample. For datasets that have Decathlon format, datalist can be
                achieved by calling `monai.data.load_decathlon_datalist`.
            image_key: the key name of images. Defaults to `image`.
            label_key: the key name of labels. Defaults to `label`.
            meta_key_postfix: for nifti images, use `{image_key}_{meta_key_postfix}` to
                store the metadata of images.
            num_workers: the maximum number of processes can be used in data loading.

        """

        self.datalist = datalist
        self.image_key = image_key
        self.label_key = label_key
        self.meta_key_postfix = meta_key_postfix
        self.num_workers = num_workers
        self.loader = LoadImaged(keys=[image_key, label_key], meta_key_postfix=meta_key_postfix)

    def _run_parallel(self, function):
        """
        Parallelly running the function for all data in the datalist.

        """

        return Parallel(n_jobs=self.num_workers)(delayed(function)(data) for data in self.datalist)

    def _load_spacing(self, path_dict: Dict):
        """
        Load spacing from a data's dictionary. Assume that the original image file has `pixdim`
        in its metadata.

        """
        data = self.loader(path_dict)
        meta_key = "{}_{}".format(self.image_key, self.meta_key_postfix)
        spacing = data[meta_key]["pixdim"][1:4].tolist()

        return spacing

    def _get_target_spacing(self, anisotropic_threshold: int = 3, percentile: float = 10.0):
        """
        Calculate the target spacing according to all spacings.
        If the target spacing is very anisotropic,
        decrease the spacing value of the maximum axis according to percentile.

        """
        spacing = self._run_parallel(self._load_spacing)
        spacing = np.array(spacing)
        target_spacing = np.median(spacing, axis=0)
        if max(target_spacing) / min(target_spacing) >= anisotropic_threshold:
            largest_axis = np.argmax(target_spacing)
            target_spacing[largest_axis] = np.percentile(spacing[:, largest_axis], percentile)

        output = list(target_spacing)
        output = [round(value, 2) for value in output]

        return tuple(output)

    def _load_intensity(self, path_dict: Dict):
        """
        Load intensity from a data's dictionary.

        """
        data = self.loader(path_dict)
        image = data[self.image_key]
        foreground_idx = np.where(data[self.label_key] > 0)

        return image[foreground_idx].tolist()

    def _get_intensity_stats(self, lower: float = 0.5, upper: float = 99.5):
        """
        Calculate min, max, mean and std of all intensities. The minimal and maximum
        values will be processed according to the provided percentiles.

        """
        intensity = self._run_parallel(self._load_intensity)
        intensity = np.array(list(itertools.chain.from_iterable(intensity)))
        min_value, max_value = np.percentile(intensity, [lower, upper])
        mean_value, std_value = np.mean(intensity), np.std(intensity)
        output = [min_value, max_value, mean_value, std_value]
        output = [round(value, 2) for value in output]

        return tuple(output)

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

import os
import sys
import tarfile
import numpy as np
from typing import Any, Callable
from monai.data import CacheDataset
from monai.transforms import Randomizable
from .utils import download_and_extract


class MedNISTDataset(Randomizable, CacheDataset):
    """
    The Dataset to automatically download MedNIST data and generate items for training, validation or test.
    It's based on `CacheDataset` to accelerate the training process.

    Args:
        root_dir: target directory to download and load MedNIST dataset.
        section: expected data section, can be: `training`, `validation` or `test`.
        transform: transforms to execute operations on input data.
        download: whether to download and extract the MedNIST from resource link, default is False.
            if expected file already exists, skip downloading even set it to True.
            user can manually copy `MedNIST.tar.gz` file or `MedNIST` folder to root directory.
        extract: whether to extract the MedNIST.tar.gz file under root directory, default is False.
        seed: random seed to randomly split training, validation and test datasets, defaut is 0.
        val_frac: percentage of of validation fraction in the whole dataset, default is 0.1.
        test_frac: percentage of of test fraction in the whole dataset, default is 0.1.
        cache_num: number of items to be cached. Default is `sys.maxsize`.
            will take the minimum of (cache_num, data_length x cache_rate, data_length).
        cache_rate: percentage of cached data in total, default is 1.0 (cache all).
            will take the minimum of (cache_num, data_length x cache_rate, data_length).
        num_workers: the number of worker threads to use.
            If 0 a single thread will be used. Default is 0.

    """

    resource = "https://www.dropbox.com/s/5wwskxctvcxiuea/MedNIST.tar.gz?dl=1"
    md5 = "0bc7306e7427e00ad1c5526a6677552d"
    compressed_file_name = "MedNIST.tar.gz"
    dataset_folder_name = "MedNIST"

    def __init__(
        self,
        root_dir: str,
        section: str,
        transform: Callable[..., Any],
        download: bool = False,
        seed: int = 0,
        val_frac: float = 0.1,
        test_frac: float = 0.1,
        cache_num: int = sys.maxsize,
        cache_rate: float = 1.0,
        num_workers: int = 0,
    ):
        if not os.path.isdir(root_dir):
            raise ValueError("root_dir must be a directory.")
        self.section = section
        self.val_frac = val_frac
        self.test_frac = test_frac
        self.set_random_state(seed=seed)
        tarfile_name = os.path.join(root_dir, self.compressed_file_name)
        dataset_dir = os.path.join(root_dir, self.dataset_folder_name)
        if download:
            download_and_extract(self.resource, tarfile_name, root_dir, self.md5)

        if not os.path.exists(dataset_dir):
            raise RuntimeError("can not find dataset directory, please use download=True to download it.")
        data = self._generate_data_list(dataset_dir)
        super().__init__(data, transform, cache_num=cache_num, cache_rate=cache_rate, num_workers=num_workers)

    def randomize(self):
        self.rann = self.R.random()

    def _generate_data_list(self, dataset_dir):
        class_names = sorted([x for x in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, x))])
        num_class = len(class_names)
        image_files = [
            [
                os.path.join(dataset_dir, class_names[i], x)
                for x in os.listdir(os.path.join(dataset_dir, class_names[i]))
            ]
            for i in range(num_class)
        ]
        num_each = [len(image_files[i]) for i in range(num_class)]
        image_files_list = []
        image_class = []
        for i in range(num_class):
            image_files_list.extend(image_files[i])
            image_class.extend([i] * num_each[i])
        num_total = len(image_class)

        data = list()

        for i in range(num_total):
            self.randomize()
            if self.section == "training":
                if self.rann < self.val_frac + self.test_frac:
                    continue
            elif self.section == "validation":
                if self.rann >= self.val_frac:
                    continue
            elif self.section == "test":
                if self.rann < self.val_frac or self.rann >= self.val_frac + self.test_frac:
                    continue
            else:
                raise ValueError("section name can only be: training, validation or test.")
            data.append({"image": image_files_list[i], "label": image_class[i]})
        return data

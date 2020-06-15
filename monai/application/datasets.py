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
import tarfile
import wget
import numpy as np
from monai.data import CacheDataset
from .utils import download_url, extractall


class MedNISTDataset(Randomizable, CacheDataset):
    resource = "https://www.dropbox.com/s/5wwskxctvcxiuea/MedNIST.tar.gz"

    def __init__(
        self, root, transform, section, download=False, cache_num=sys.maxsize, cache_rate=1.0, num_workers=0, seed=0
    ):
        if not os.path.isdir(root):
            raise ValueError("root must be a directory.")
        self.root = root
        self.section = section
        self.set_random_state(seed=seed)
        self.tarfile_name = os.path.join(self.root, "MedNIST.tar.gz")
        self.dataset_dir = os.path.join(self.root, "MedNIST")
        if download:
            self._download()
        if os.path.exists(self.tarfile_name) and not os.path.exists(self.dataset_dir):
            extractall()
        if not os.path.exists(self.dataset_dir):
            raise RuntimeError("can not find dataset directory, please use download=True to download it.")
        data = self._generate_data_list()
        super().__init__(data, transform, cache_num=cache_num, cache_rate=cache_rate, num_workers=num_workers)

    def randomize(self):
        self.rann = self.R.random()

    def _download(self):
        if os.path.exists(self.tarfile_name) or os.path.exists(self.dataset_dir):
            return
        os.makedirs(self.root, exist_ok=True)
        download_url(self.resource, self.tarfile_name)

    def _generate_data_list(self):
        class_names = sorted(
            [x for x in os.listdir(self.dataset_dir) if os.path.isdir(os.path.join(self.dataset_dir, x))]
        )
        num_class = len(class_names)
        image_files = [
            [
                os.path.join(self.dataset_dir, class_names[i], x)
                for x in os.listdir(os.path.join(self.dataset_dir, class_names[i]))
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

        val_frac = 0.1
        test_frac = 0.1
        data = list()

        for i in range(num_total):
            self.randomize()
            if self.section == "training":
                if self.rann < val_frac + test_frac:
                    continue
            elif self.section == "validation":
                if self.rann >= val_frac:
                    continue
            elif self.section == "test":
                if self.rann < val_frac or self.rann >= val_frac + test_frac:
                    continue
            else:
                raise ValueError("section name can only be: training, validation or test.")
            data.append({"image": image_files_list[i], "label": image_class[i]})
        return data

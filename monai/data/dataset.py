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

import sys
import torch
from monai.transforms.compose import Compose
from monai.utils import process_bar


class Dataset(torch.utils.data.Dataset):
    """
    Generic dataset to handle dictionary format data, it can operate transforms for specific fields.
    For example, typical input data can be a list of dictionaries::

        [{                            {                            {
             'img': 'image1.nii.gz',      'img': 'image2.nii.gz',      'img': 'image3.nii.gz',
             'seg': 'label1.nii.gz',      'seg': 'label2.nii.gz',      'seg': 'label3.nii.gz',
             'extra': 123                 'extra': 456                 'extra': 789
         },                           },                           }]
    """

    def __init__(self, data, transform=None):
        """
        Args:
            data (Iterable): input data to load and transform to generate dataset for model.
            transform (Callable, optional): transforms to excute operations on input data.
        """
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        if self.transform is not None:
            data = self.transform(data)

        return data


class CacheDataset(Dataset):
    """
    Dataset with cache mechanism that loads data and runs deterministic transforms before training.
    During the training process, it will skip the deterministic transforms and load data from cache
    for the random transforms directly, so it's much faster than repeatedly running these operations
    for every epoch. If the data is not in cache, run all the transforms directly.
    And users can set the cache rate or cache data number according to the memory size.
    """

    def __init__(self, data, transform, cache_num=sys.maxsize, cache_rate=1.0):
        """
        Args:
            data (Iterable): input data to load and transform to generate dataset for model.
            transform (Callable, optional): transforms to excute operations on input data.
            cache_num (int): number of cached items, default is infinity.
                will take the minimum of (cache_num, data_length x cache_rate, data_length).
            cache_rate (float): percentage of cached data in total, default is 1.0(cache all).
                will take the minimum of (cache_num, data_length x cache_rate, data_length).
        """
        assert isinstance(transform, Compose), 'CacheDataset expects Compose transform.'
        super().__init__(data, transform)
        self.cache_num = min(cache_num, int(len(data) * cache_rate), len(data))
        self._cache = list()
        print('Loading data to cache and executing transforms...')
        for i in range(self.cache_num):
            process_bar(i + 1, self.cache_num)
            self._cache.append(transform(data[i], deterministic=True))

    def __getitem__(self, index):
        if index < self.cache_num:
            data = self.transform(self._cache[index], deterministic=False)
        else:
            data = self.transform(self.data[index], deterministic=None)

        return data

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
from monai.transforms.compose import Randomizable


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

    And users can set the cache rate or cache data number according to the memory size, suggest to set
    the `cache_num` or `cache_rate` to the biggest value to achieve best training performance.

    To improve the caching efficiency, please always put as many as possible non-random transforms
    before the randomised ones when composing the chain of transforms.

    For example, if define the training transforms as:
    transforms = Compose([
        LoadNiftid(),
        AddChanneld(),
        Spacingd(),
        Orientationd(),
        ScaleIntensityRanged(),
        RandCropByPosNegLabeld(),
        ToTensord()
    ])
    Before training, run: `LoadNiftid`, `AddChanneld`, `Spacingd`, `Orientationd`, `ScaleIntensityRanged`.
    During training, load cache and run: `RandCropByPosNegLabeld`, `ToTensord`.
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
            item = data[i]
            for _transform in transform.transforms:
                # execute all the deterministic transforms before the first random transform
                if isinstance(_transform, Randomizable):
                    break
                item = transform.do_transform(item, _transform)
            self._cache.append(item)

    def __getitem__(self, index):
        if index < self.cache_num:
            # load data from cache and execute from the first random transform
            start_run = False
            data = self._cache[index]
            for _transform in self.transform.transforms:
                if not start_run and not isinstance(_transform, Randomizable):
                    continue
                else:
                    start_run = True
                data = self.transform.do_transform(data, _transform)
        else:
            # if didn't cache this data, execute all the transforms directly
            data = self.transform(self.data[index])

        return data

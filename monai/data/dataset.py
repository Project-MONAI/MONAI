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

from multiprocessing.pool import ThreadPool
import threading

from monai.transforms.compose import Compose, Randomizable
from monai.transforms.utils import apply_transform
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
            transform (Callable, optional): transforms to execute operations on input data.
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
    Dataset with cache mechanism that can load data and cache deterministic transforms' result during training.

    By caching the results of non-random preprocessing transforms, it accelerates the training data pipeline.
    If the requested data is not in the cache, all transforms will run normally
    (see also :py:class:`monai.data.dataset.Dataset`).

    Users can set the cache rate or number of items to cache.
    It is recommended to experiment with different `cache_num` or `cache_rate` to identify the best training speed.

    To improve the caching efficiency, please always put as many as possible non-random transforms
    before the randomised ones when composing the chain of transforms.

    For example, if the transform is a `Compose` of::

        transforms = Compose([
            LoadNiftid(),
            AddChanneld(),
            Spacingd(),
            Orientationd(),
            ScaleIntensityRanged(),
            RandCropByPosNegLabeld(),
            ToTensord()
        ])

    when `transforms` is used in a multi-epoch training pipeline, before the first training epoch,
    this dataset will cache the results up to ``ScaleIntensityRanged``, as
    all non-random transforms `LoadNiftid`, `AddChanneld`, `Spacingd`, `Orientationd`, `ScaleIntensityRanged`
    can be cached. During training, the dataset will load the cached results and run
    ``RandCropByPosNegLabeld`` and ``ToTensord``, as ``RandCropByPosNegLabeld`` is a randomised transform
    and the outcome not cached.
    """

    def __init__(self, data, transform, cache_num=sys.maxsize, cache_rate=1.0, num_workers=0):
        """
        Args:
            data (Iterable): input data to load and transform to generate dataset for model.
            transform (Callable): transforms to execute operations on input data.
            cache_num (int): number of items to be cached. Default is `sys.maxsize`.
                will take the minimum of (cache_num, data_length x cache_rate, data_length).
            cache_rate (float): percentage of cached data in total, default is 1.0 (cache all).
                will take the minimum of (cache_num, data_length x cache_rate, data_length).
            num_workers (int): the number of worker processes to use.
                If 0 a single thread will be used. Default is 0.
        """
        if not isinstance(transform, Compose):
            transform = Compose(transform)
        super().__init__(data, transform)
        self.cache_num = min(cache_num, int(len(self) * cache_rate), len(self))
        self._cache = [None] * self.cache_num
        print('Load and cache transformed data...')
        if num_workers > 0:
            self._item_processed = 0
            self._thread_lock = threading.Lock()
            with ThreadPool(num_workers) as p:
                p.map(self._load_cache_item_thread, [(i, data[i], transform.transforms) for i in range(self.cache_num)])
        else:
            for i in range(self.cache_num):
                self._cache[i] = self._load_cache_item(data[i], transform.transforms)
                process_bar(i + 1, self.cache_num)

    def _load_cache_item(self, item, transforms):
        for _transform in transforms:
            # execute all the deterministic transforms before the first random transform
            if isinstance(_transform, Randomizable):
                break
            item = apply_transform(_transform, item)
        return item

    def _load_cache_item_thread(self, args):
        i, item, transforms = args
        self._cache[i] = self._load_cache_item(item, transforms)
        with self._thread_lock:
            self._item_processed += 1
            process_bar(self._item_processed, self.cache_num)

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
                data = apply_transform(_transform, data)
        else:
            # no cache for this data, execute all the transforms directly
            data = super(CacheDataset, self).__getitem__(index)
        return data

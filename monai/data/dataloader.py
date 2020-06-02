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

import torch
from monai.data import CacheDataset, PersistentDataset, list_data_collate


class ImageDataLoader(torch.utils.data.DataLoader):
    """Generates images/labels for train/validation/testing according to the datalist.
    Expects the images/labels and transforms are for dictionary format data.
    It inherits from PyTorch DataLoader and adds callbacks for `collate` and `worker_fn`.
    Also supports CacheDataset and PersistDataset, so common users can use this DataLoader directly.

    Args:
        datalist (list of dict): data source for Dataset, expect to be a list of dict object, for example:
            `[{'image': 'chest_19.nii.gz',  'label': 0}, {'image': 'chest_31.nii.gz',  'label': 1}]`.
        transform (Transform): transform to be applied to the data.
            typically, a list of transforms composed by `Compose`.
        batch_size (int): batch size of the DataLoader output.
        num_workers (int): number of worker processes for data transformation.
        shuffle (bool): whether to shuffle the data or not.
        cache_num (int): number of items to be cached in memory. default is 0, disabled cache.
            will take the minimum of (cache_num, data_length x cache_rate, data_length).
        cache_rate (float): percentage of cached data in total in memory, default is 0, disabled cache.
            will take the minimum of (cache_num, data_length x cache_rate, data_length).
        cache_dir (str): if set the directory, it will cache all the data into file and load for every epoch.
            `cache_num` and `cache_rate` will be disabled in this case.

    """
    def __init__(
        self,
        datalist,
        transform,
        batch_size=1,
        num_workers=0,
        shuffle=False,
        cache_num=0,
        cache_rate=0,
        cache_dir=None
    ):
        if not isinstance(datalist, (list, tuple)):
            raise ValueError('datalist must be a list of dictionary objects.')
        for i in datalist:
            if not isinstance(i, dict):
                raise ValueError('items of datalist must be dictionary objects.')

        def worker_init_fn(worker_id):
            worker_info = torch.utils.data.get_worker_info()
            worker_info.dataset.transform.set_random_state(worker_info.seed % (2 ** 32))
        if isinstance(cache_dir, str):
            dataset = PersistentDataset(data=datalist, transform=transform, cache_dir=cache_dir)
        else:
            dataset = CacheDataset(data=datalist, transform=transform, cache_num=cache_num, cache_rate=cache_rate)

        super().__init__(
            dataset=dataset,
            collate_fn=list_data_collate,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            worker_init_fn=worker_init_fn
        )

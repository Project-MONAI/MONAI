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
import random
from typing import Callable
from monai.data import CacheDataset

class RGDataset(CacheDataset):
    """
    Wraps CacheDataset with custom functionality for Radiogenomic GAN training.

    RGGAN requires an adversarial datapoint of a different protein label to enforce disentangled RNA mapping.

    This additional datapoint can be seen as variables wrong_imgs and wrong_segs below.
    """
    def __init__(
        self, data, transform: Callable, cache_num: int = sys.maxsize, cache_rate: float = 1.0, num_workers: int = 0
    ):
        """
        Args:
            data (Iterable): input data to load and transform to generate dataset for model.
            transform: transforms to execute operations on input data.
            cache_num: number of items to be cached. Default is `sys.maxsize`.
                will take the minimum of (cache_num, data_length x cache_rate, data_length).
            cache_rate: percentage of cached data in total, default is 1.0 (cache all).
                will take the minimum of (cache_num, data_length x cache_rate, data_length).
            num_workers: the number of worker threads to use.
                If 0 a single thread will be used. Default is 0.
        """
        super().__init__(data, transform, cache_num, cache_rate, num_workers)
        
    def __getitem__(self, index):
        datapoint = super(CacheDataset, self).__getitem__(index)
        
        # imgs = datapoint['image']
        # segs = datapoint['seg']
        # embedding = datapoint['embedding']
        # base_img_name = datapoint['base']
        
        length = self.__len__()
        rand_index = random.randint(0, length - 1)
        wrong_datapoint = super(CacheDataset, self).__getitem__(rand_index)

        recursion_failsafe = 0
        while wrong_datapoint['class'] == datapoint['class'] and recursion_failsafe < 10:
            rand_index = random.randint(0, length - 1)
            wrong_datapoint = super(CacheDataset, self).__getitem__(rand_index)
            recursion_failsafe += 1

        datapoint['w_image'] = wrong_datapoint['image']
        datapoint['w_seg'] = wrong_datapoint['seg']
        
        return datapoint
        #return [imgs], [segs], [wrong_imgs], [wrong_segs], embedding, base_img_name

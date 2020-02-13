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


import math

import torch
from torch.utils.data import IterableDataset

from monai.utils.module import export
from monai.data.utils import iter_patch


@export("monai.data")
class GridPatchDataset(IterableDataset):
    """
    Yields patches from arrays read from an input dataset. The patches are chosen in a contiguous grid sampling scheme.
    """

    def __init__(self, dataset, patch_size, start_pos=(), pad_mode="wrap", **pad_opts):
        """
        Initializes this dataset in terms of the input dataset and patch size. The `patch_size` is the size of the 
        patch to sample from the input arrays. Tt is assumed the arrays first dimension is the channel dimension which
        will be yielded in its entirety so this should not be specified in `patch_size`. For example, for an input 3D
        array with 1 channel of size (1, 20, 20, 20) a regular grid sampling of eight patches (1, 10, 10, 10) would be 
        specified by a `patch_size` of (10, 10, 10).

        Args:
            dataset (Dataset): the dataset to read array data from
            patch_size (tuple of int or None): size of patches to generate slices for, 0/None selects whole dimension
            start_pos (tuple of it, optional): starting position in the array, default is 0 for each dimension
            pad_mode (str, optional): padding mode, see numpy.pad
            pad_opts (dict, optional): padding options, see numpy.pad
        """

        self.dataset = dataset
        self.patch_size = (None,) + tuple(patch_size)
        self.start_pos = start_pos
        self.pad_mode = pad_mode
        self.pad_opts = pad_opts

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        iter_start = 0
        iter_end = len(self.dataset)

        if worker_info is not None:  
            # split workload
            per_worker = int(math.ceil((iter_end - iter_start) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = iter_start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, iter_end)

        for index in range(iter_start, iter_end):
            arrays = self.dataset[index]

            iters = [iter_patch(a, self.patch_size, self.start_pos, False, self.pad_mode, **self.pad_opts) for a in arrays]

            yield from zip(*iters)

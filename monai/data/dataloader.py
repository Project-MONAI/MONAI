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

import torch
from torch.utils.data import DataLoader as _TorchDataLoader
from torch.utils.data import Dataset

from monai.data.utils import list_data_collate, set_rnd, worker_init_fn

__all__ = ["DataLoader"]


class DataLoader(_TorchDataLoader):
    """Generates images/labels for train/validation/testing from dataset.
    It inherits from PyTorch DataLoader and adds default callbacks for `collate`
    and `worker_fn` if user doesn't set them.

    More information about PyTorch DataLoader, please check:
    https://github.com/pytorch/pytorch/blob/master/torch/utils/data/dataloader.py

    Args:
        dataset: dataset from which to load the data.
        num_workers: how many subprocesses to use for data
            loading. ``0`` means that the data will be loaded in the main process.
            (default: ``0``)
        kwargs: other parameters for PyTorch DataLoader.

    """

    def __init__(self, dataset: Dataset, num_workers: int = 0, **kwargs) -> None:
        if num_workers == 0:
            # when num_workers > 0, random states are determined by worker_init_fn
            # this is to make the behavior consistent when num_workers == 0
            # torch.int64 doesn't work well on some versions of windows
            _seed = torch.empty((), dtype=torch.int32).random_(generator=None).item()
            set_rnd(dataset, int(_seed))
        if "collate_fn" not in kwargs:
            kwargs.update({"collate_fn": list_data_collate})
        if "worker_init_fn" not in kwargs:
            kwargs.update({"worker_init_fn": worker_init_fn})

        super().__init__(  # type: ignore[call-overload]
            dataset=dataset,
            num_workers=num_workers,
            **kwargs,
        )

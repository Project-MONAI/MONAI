# Copyright (c) MONAI Consortium
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
    """
    Provides an iterable over the given `dataset`.  It inherits the PyTorch
    DataLoader and adds enhanced `collate_fn` and `worker_fn` by default.

    Although this class could be configured to be the same as
    `torch.utils.data.DataLoader`, its default configuration is
    recommended, mainly for the following extra features:

        - It handles MONAI randomizable objects with appropriate random state
          managements for deterministic behaviour.
        - It is aware of the patch-based transform (such as
          :py:class:`monai.transforms.RandSpatialCropSamplesDict`) samples for
          preprocessing with enhanced data collating behaviour.
          See: :py:class:`monai.transforms.Compose`.

    For more details about :py:class:`torch.utils.data.DataLoader`, please see:
    https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader.

    For example, to construct a randomized dataset and iterate with the data loader:

    .. code-block:: python

        import torch

        from monai.data import DataLoader
        from monai.transforms import Randomizable


        class RandomDataset(torch.utils.data.Dataset, Randomizable):
            def __getitem__(self, index):
                return self.R.randint(0, 1000, (1,))

            def __len__(self):
                return 16


        dataset = RandomDataset()
        dataloader = DataLoader(dataset, batch_size=2, num_workers=4)
        for epoch in range(2):
            for i, batch in enumerate(dataloader):
                print(epoch, i, batch.data.numpy().flatten().tolist())

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

        super().__init__(dataset=dataset, num_workers=num_workers, **kwargs)  # type: ignore[call-overload]

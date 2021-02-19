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

from typing import Callable, Optional
from monai.data.dataloader import DataLoader
from torch.utils.data.dataloader import DataLoader as TorchDataLoader
from monai.data.dataset import Dataset
from monai.data.utils import decollate_batch
from monai.transforms.inverse_transform import InvertibleTransform

__all__ = ["BatchInverseTransform"]


class _BatchInverseDataset(Dataset):
    def __init__(self, data, transform: InvertibleTransform) -> None:
        self.data = decollate_batch(data)
        self.transform = transform

    def __getitem__(self, index: int):
        data = self.data[index]
        return self.transform.inverse(data)


class BatchInverseTransform:
    """something"""

    def __init__(
        self, transform: InvertibleTransform, loader: TorchDataLoader, collate_fn: Optional[Callable] = None
    ) -> None:
        """
        Args:
            transform: a callable data transform on input data.
            loader: data loader used to generate the batch of data.
            collate_fn: how to collate data after inverse transformations. Default will use the DataLoader's default collation method.
                If returning images of different sizes, this will likely create an error (since the collation will concatenate arrays,
                requiring them to be the same size). In this case, using `collate_fn=lambda x: x` might solve the problem.
        """
        self.transform = transform
        self.batch_size = loader.batch_size
        self.num_workers = loader.num_workers
        self.collate_fn = collate_fn

    def __call__(self, data):
        inv_ds = _BatchInverseDataset(data, self.transform)
        inv_loader = DataLoader(inv_ds, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn)
        try:
            return next(iter(inv_loader))
        except RuntimeError as re:
            re_str = str(re)
            if "stack expects each tensor to be equal size" in re_str:
                re_str += "\nMONAI hint: try creating `BatchInverseTransform` with `collate_fn=lambda x: x`."
            raise RuntimeError(re_str)

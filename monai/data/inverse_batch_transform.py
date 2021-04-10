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

import warnings
from typing import Any, Callable, Dict, Hashable, Optional, Sequence

import numpy as np
from torch.utils.data.dataloader import DataLoader as TorchDataLoader

from monai.data.dataloader import DataLoader
from monai.data.dataset import Dataset
from monai.data.utils import decollate_batch, no_collation, pad_list_data_collate
from monai.transforms.croppad.batch import PadListDataCollate
from monai.transforms.inverse import InvertibleTransform
from monai.transforms.transform import Transform
from monai.utils import first

__all__ = ["BatchInverseTransform"]


class _BatchInverseDataset(Dataset):
    def __init__(
        self,
        data: Sequence[Any],
        transform: InvertibleTransform,
        pad_collation_used: bool,
    ) -> None:
        super().__init__(data, transform)
        self.invertible_transform = transform
        self.pad_collation_used = pad_collation_used

    def _transform(self, index: int) -> Dict[Hashable, np.ndarray]:
        data = dict(self.data[index])
        # If pad collation was used, then we need to undo this first
        if self.pad_collation_used:
            data = PadListDataCollate.inverse(data)

        if not isinstance(self.invertible_transform, InvertibleTransform):
            warnings.warn("transform is not invertible, can't invert transform for the input data.")
            return data
        return self.invertible_transform.inverse(data)


class BatchInverseTransform(Transform):
    """
    Perform inverse on a batch of data. This is useful if you have inferred a batch of images and want to invert
    them all.
    """

    def __init__(
        self,
        transform: InvertibleTransform,
        loader: TorchDataLoader,
        collate_fn: Optional[Callable] = no_collation,
        num_workers: Optional[int] = 0,
    ) -> None:
        """
        Args:
            transform: a callable data transform on input data.
            loader: data loader used to run `transforms` and generate the batch of data.
            collate_fn: how to collate data after inverse transformations.
                default won't do any collation, so the output will be a list of size batch size.
            num_workers: number of workers when run data loader for inverse transforms,
                default to 0 as only run 1 iteration and multi-processing may be even slower.
                if the transforms are really slow, set num_workers for multi-processing.
                if set to `None`, use the `num_workers` of the transform data loader.
        """
        self.transform = transform
        self.batch_size = loader.batch_size
        self.num_workers = loader.num_workers if num_workers is None else num_workers
        self.collate_fn = collate_fn
        self.pad_collation_used = loader.collate_fn == pad_list_data_collate

    def __call__(self, data: Dict[str, Any]) -> Any:

        decollated_data = decollate_batch(data)
        inv_ds = _BatchInverseDataset(decollated_data, self.transform, self.pad_collation_used)
        inv_loader = DataLoader(
            inv_ds, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn
        )
        try:
            return first(inv_loader)
        except RuntimeError as re:
            re_str = str(re)
            if "equal size" in re_str:
                re_str += "\nMONAI hint: try creating `BatchInverseTransform` with `collate_fn=lambda x: x`."
            raise RuntimeError(re_str)

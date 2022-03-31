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

import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader as TorchDataLoader

from monai.config import KeysCollection
from monai.data.dataloader import DataLoader
from monai.data.utils import decollate_batch, no_collation, pad_list_data_collate
from monai.transforms.croppad.batch import PadListDataCollate
from monai.transforms.inverse import InvertibleTransform
from monai.transforms.transform import MapTransform, Transform
from monai.utils import first

__all__ = ["BatchInverseTransform", "Decollated", "DecollateD", "DecollateDict"]


class _BatchInverseDataset(Dataset):
    def __init__(self, data: Sequence[Any], transform: InvertibleTransform, pad_collation_used: bool) -> None:
        self.data = data
        self.invertible_transform = transform
        self.pad_collation_used = pad_collation_used

    def __getitem__(self, index: int):
        data = dict(self.data[index])
        # If pad collation was used, then we need to undo this first
        if self.pad_collation_used:
            data = PadListDataCollate.inverse(data)

        if not isinstance(self.invertible_transform, InvertibleTransform):
            warnings.warn("transform is not invertible, can't invert transform for the input data.")
            return data
        return self.invertible_transform.inverse(data)

    def __len__(self) -> int:
        return len(self.data)


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
        detach: bool = True,
        pad_batch: bool = True,
        fill_value=None,
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
            detach: whether to detach the tensors. Scalars tensors will be detached into number types
                instead of torch tensors.
            pad_batch: when the items in a batch indicate different batch size,
                whether to pad all the sequences to the longest.
                If False, the batch size will be the length of the shortest sequence.
            fill_value: the value to fill the padded sequences when `pad_batch=True`.

        """
        self.transform = transform
        self.batch_size = loader.batch_size
        self.num_workers = loader.num_workers if num_workers is None else num_workers
        self.collate_fn = collate_fn
        self.detach = detach
        self.pad_batch = pad_batch
        self.fill_value = fill_value
        self.pad_collation_used = loader.collate_fn.__doc__ == pad_list_data_collate.__doc__

    def __call__(self, data: Dict[str, Any]) -> Any:
        decollated_data = decollate_batch(data, detach=self.detach, pad=self.pad_batch, fill_value=self.fill_value)
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
            raise RuntimeError(re_str) from re


class Decollated(MapTransform):
    """
    Decollate a batch of data. If input is a dictionary, it also supports to only decollate specified keys.
    Note that unlike most MapTransforms, it will delete the other keys that are not specified.
    if `keys=None`, it will decollate all the data in the input.
    It replicates the scalar values to every item of the decollated list.

    Args:
        keys: keys of the corresponding items to decollate, note that it will delete other keys not specified.
            if None, will decollate all the keys. see also: :py:class:`monai.transforms.compose.MapTransform`.
        detach: whether to detach the tensors. Scalars tensors will be detached into number types
            instead of torch tensors.
        pad_batch: when the items in a batch indicate different batch size,
            whether to pad all the sequences to the longest.
            If False, the batch size will be the length of the shortest sequence.
        fill_value: the value to fill the padded sequences when `pad_batch=True`.
        allow_missing_keys: don't raise exception if key is missing.

    """

    def __init__(
        self,
        keys: Optional[KeysCollection] = None,
        detach: bool = True,
        pad_batch: bool = True,
        fill_value=None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.detach = detach
        self.pad_batch = pad_batch
        self.fill_value = fill_value

    def __call__(self, data: Union[Dict, List]):
        d: Union[Dict, List]
        if len(self.keys) == 1 and self.keys[0] is None:
            # it doesn't support `None` as the key
            d = data
        else:
            if not isinstance(data, dict):
                raise TypeError("input data is not a dictionary, but specified keys to decollate.")
            d = {}
            for key in self.key_iterator(data):
                d[key] = data[key]

        return decollate_batch(d, detach=self.detach, pad=self.pad_batch, fill_value=self.fill_value)


DecollateD = DecollateDict = Decollated

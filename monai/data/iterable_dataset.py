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

from typing import Callable, Iterable, Optional

from torch.utils.data import IterableDataset as _TorchIterableDataset

from monai.transforms import apply_transform


class IterableDataset(_TorchIterableDataset):
    """
    A generic dataset for iterable data source and an optional callable data transform
    when fetching a data sample.
    For example, typical input data can be web data stream which can support multi-process access.

    Note that when used with `DataLoader` and `num_workers > 0`, each worker process will have a
    different copy of the dataset object, need to guarantee process-safe from data source or DataLoader.

    """

    def __init__(self, data: Iterable, transform: Optional[Callable] = None) -> None:
        """
        Args:
            data: input data source to load and transform to generate dataset for model.
            transform: a callable data transform on input data.
        """
        self.data = data
        self.transform = transform
        self.source = None

    def __iter__(self):
        self.source = iter(self.data)
        for data in self.source:
            if self.transform is not None:
                data = apply_transform(self.transform, data)
            yield data

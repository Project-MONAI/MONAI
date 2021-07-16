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

import math
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Union

from torch.utils.data import IterableDataset as _TorchIterableDataset
from torch.utils.data import get_worker_info

from monai.data.utils import convert_tables_to_dicts
from monai.transforms import apply_transform
from monai.utils import ensure_tuple, optional_import

pd, _ = optional_import("pandas")


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


class CSVIterableDataset(IterableDataset):
    """
    Iterable dataset to load CSV files and generate dictionary data.
    It can be helpful when loading extremely big CSV files that can't read into memory directly.
    To accelerate the loading process, it can support multi-processing based on PyTorch DataLoader workers,
    every process executes transforms on part of every loaded chunk.
    Note: the order of output data may not match data source in multi-processing mode.

    It can load data from multiple CSV files and join the tables with additional `kwargs` arg.
    Support to only load specific columns.
    And it can also group several loaded columns to generate a new column, for example,
    set `col_groups={"meta": ["meta_0", "meta_1", "meta_2"]}`, output can be::

        [
            {"image": "./image0.nii", "meta_0": 11, "meta_1": 12, "meta_2": 13, "meta": [11, 12, 13]},
            {"image": "./image1.nii", "meta_0": 21, "meta_1": 22, "meta_2": 23, "meta": [21, 22, 23]},
        ]

    Args:
        filename: the filename of expected CSV file to load. if providing a list
            of filenames, it will load all the files and join tables.
        chunksize: rows of a chunk when loading iterable data from CSV files, default to 1000. more details:
            https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html.
        col_names: names of the expected columns to load. if None, load all the columns.
        col_types: `type` and `default value` to convert the loaded columns, if None, use original data.
            it should be a dictionary, every item maps to an expected column, the `key` is the column
            name and the `value` is None or a dictionary to define the default value and data type.
            the supported keys in dictionary are: ["type", "default"]. for example::

                col_types = {
                    "subject_id": {"type": str},
                    "label": {"type": int, "default": 0},
                    "ehr_0": {"type": float, "default": 0.0},
                    "ehr_1": {"type": float, "default": 0.0},
                    "image": {"type": str, "default": None},
                }

        col_groups: args to group the loaded columns to generate a new column,
            it should be a dictionary, every item maps to a group, the `key` will
            be the new column name, the `value` is the names of columns to combine. for example:
            `col_groups={"ehr": [f"ehr_{i}" for i in range(10)], "meta": ["meta_1", "meta_2"]}`
        transform: transform to apply on the loaded items of a dictionary data.
        kwargs: additional arguments for `pandas.merge()` API to join tables.

    """

    def __init__(
        self,
        filename: Union[str, Sequence[str]],
        chunksize: int = 1000,
        col_names: Optional[Sequence[str]] = None,
        col_types: Optional[Dict[str, Optional[Dict[str, Any]]]] = None,
        col_groups: Optional[Dict[str, Sequence[str]]] = None,
        transform: Optional[Callable] = None,
        **kwargs,
    ):
        self.files = ensure_tuple(filename)
        self.chunksize = chunksize
        self.iters = self.reset()
        self.col_names = col_names
        self.col_types = col_types
        self.col_groups = col_groups
        self.kwargs = kwargs
        super().__init__(data=None, transform=transform)  # type: ignore

    def reset(self, filename: Optional[Union[str, Sequence[str]]] = None):
        if filename is not None:
            # update files if necessary
            self.files = ensure_tuple(filename)
        self.iters = [pd.read_csv(f, chunksize=self.chunksize) for f in self.files]
        return self.iters

    def __iter__(self):
        for chunks in zip(*self.iters):
            self.data = convert_tables_to_dicts(
                dfs=chunks,
                col_names=self.col_names,
                col_types=self.col_types,
                col_groups=self.col_groups,
                **self.kwargs,
            )
            info = get_worker_info()
            if info is not None:
                length = len(self.data)
                per_worker = int(math.ceil(length / float(info.num_workers)))
                start = info.id * per_worker
                self.data = self.data[start : min(start + per_worker, length)]

            return super().__iter__()

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

from functools import reduce
from typing import Dict, List, Optional, Sequence, Union

from monai.utils import ensure_tuple, optional_import

pd, _ = optional_import("pandas")


def load_csv_datalist(
    filename: Union[str, Sequence[str]],
    row_indices: Optional[Sequence[Union[int, str]]] = None,
    col_names: Optional[Sequence[str]] = None,
    col_groups: Optional[Dict[str, Sequence[str]]] = None,
    **kwargs,
) -> List[Dict]:
    """
    Utility to load data from CSV files and return a list of dictionaries,
    every dictionay maps to a row of the CSV file, and the keys of dictionary
    map to the column names of the CSV file.

    It can load multiple CSV files and join the tables with addtional `kwargs`.
    To support very big CSV files, it can load specific rows and columns. And it
    can also group several loaded columns to generate a new column, for example,
    set `col_groups={"meta": ["meta_0", "meta_1", "meta_2"]}`, output can be::

        [
            {"image": "./image0.nii", "meta_0": 11, "meta_1": 12, "meta_2": 13, "meta": [11, 12, 13]},
            {"image": "./image1.nii", "meta_0": 21, "meta_1": 22, "meta_2": 23, "meta": [21, 22, 23]},
        ]

    Args:
        filename: the filename of expected CSV file to load. if providing a list
            of filenames, it will load all the files and join tables.
        row_indices: indices of the expected rows to load. it should be a list,
            every item can be a int number or a range `[start, end)` for the indices.
            for example: `row_indices=[[0, 100], 200, 201, 202, 300]`. if None,
            load all the rows.
        col_names: names of the expected columns to load. if None, load all the columns.
        col_groups: args to group the loaded columns to generate a new column,
            it should be a dictionary, every item maps to a group, the `key` will
            be the new column name, the `value` is the names of columns to combine.
        kwargs: additional arguments for `pandas.merge()` API to join tables.

    """
    files = ensure_tuple(filename)
    # join tables with additional kwargs
    dfs = [pd.read_csv(f) for f in files]
    df = reduce(lambda l, r: pd.merge(l, r, **kwargs), dfs)

    # parse row indices
    rows: List[Union[int, str]] = []
    if row_indices is None:
        rows = list(range(df.shape[0]))
    else:
        for i in row_indices:
            if isinstance(i, (tuple, list)):
                if len(i) != 2:
                    raise ValueError("range of row indices must contain 2 values: start and end.")
                rows.extend(list(range(i[0], i[1])))
            else:
                rows.append(i)

    # convert to a list of dictionaries corresponding to every row
    data: List[Dict] = (df.loc[rows] if col_names is None else df.loc[rows, col_names]).to_dict(orient="records")

    # group columns to generate new column
    if col_groups is not None:
        groups: Dict[str, List] = {}
        for name, cols in col_groups.items():
            groups[name] = df.loc[rows, cols].values
        # invert items of groups to every row of data
        data = [dict(d, **{k: v[i] for k, v in groups.items()}) for i, d in enumerate(data)]

    return data

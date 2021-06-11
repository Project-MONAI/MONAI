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
    """Load data list from CSV files.

    """
    files = ensure_tuple(filename)
    # join tables with additional kwargs
    dfs = [pd.read_csv(f) for f in files]
    df = reduce(lambda l, r: pd.merge(l, r, **kwargs), dfs)

    # parse row indices
    rows: List[int]
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

    data = df.loc(rows) if col_names is None else df.loc(rows, col_names)

    # group columns to generate new column
    if col_groups is not None:
        for name, cols in col_names.items():
            data[name] = df.loc(rows, cols).values

    # convert to a list of dictionaries
    length = len(data[data.keys()[0]])
    return [{data[k][i] for k in data.keys()} for i in length]

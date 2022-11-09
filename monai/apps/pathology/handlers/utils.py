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

from monai.config import KeysCollection
from monai.utils import ensure_tuple


def from_engine_hovernet(keys: KeysCollection, nested_key: str):
    """
    Since the output of HoVerNet is a dictionary, this function is to extend `monai.handlers.from_engine`
    to work with HoVerNet. 

    If data is a list of nestes dictionaries after decollating, extract nested value with expected keys and 
    construct lists respectively, for example,
    if data is `[{"A": {"C": 1, "D": 2}, "B": {"C": 2, "D": 2}}, {"A":  {"C": 3, "D": 2}, "B":  {"C": 4, "D": 2}}]`,
    from_engine_hovernet(["A", "B"], "C"): `([1, 3], [2, 4])`.

    Here is a simple example::

        from monai.handlers import MeanDice, from_engine_hovernet

        metric = MeanDice(
            include_background=False,
            output_transform=from_engine_hovernet(keys=["pred", "label"], nested_key=HoVerNetBranch.NP.value)
        )

    Args:
        keys: specified keys to extract data from dictionary or decollated list of dictionaries.
        nested_key: specified key to extract nested data from dictionary or decollated list of dictionaries.

    """
    keys = ensure_tuple(keys)

    def _wrapper(data):
        if isinstance(data, dict):
            return tuple(data[k][nested_key] for k in keys)
        if isinstance(data, list) and isinstance(data[0], dict):
            # if data is a list of dictionaries, extract expected keys and construct lists,
            ret = [[i[k][nested_key] for i in data] for k in keys]
            return tuple(ret) if len(ret) > 1 else ret[0]

    return _wrapper

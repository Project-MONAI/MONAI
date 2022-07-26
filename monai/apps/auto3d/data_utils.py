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

import json
import os
from typing import Dict, Union

__all__ = ["recursive_getvalue", "recursive_setvalue", "recursive_getkey", "datafold_read"]


def datafold_read(datalist: Union[str, Dict], basedir: str, fold: int = 0, key: str = "training"):
    """

    :param datalist: the name of a JSON file listing the data, or a dictionary
    :param basedir: directory of json file
    :param fold: which fold to use (0..1 if in training set)
    :param key: usually 'training' , but can try 'validation' or 'testing' to get the list data without labels (used in challenges)
    :return:  our own 2 arrays (training, validation)
    """

    if isinstance(datalist, str):
        with open(datalist) as f:
            json_data = json.load(f)
    else:
        json_data = datalist

    json_data = json_data[key]

    for d in json_data:
        for k, v in d.items():
            if isinstance(d[k], list):
                d[k] = [os.path.join(basedir, iv) for iv in d[k]]
            elif isinstance(d[k], str):
                d[k] = os.path.join(basedir, d[k]) if len(d[k]) > 0 else d[k]

    tr = []
    val = []
    for d in json_data:
        if "fold" in d and d["fold"] == fold:
            val.append(d)
        else:
            tr.append(d)

    return tr, val


def recursive_getvalue(dicts, keys):
    """Get value of key chain. For exmaple, dicts = {'a':{'b':{'c':1}}}, keys = ['a','b','c']
    recursive_getvalue(dicts, keys) = 1
    """
    if keys[0] in dicts.keys():
        return dicts.get(keys[0]) if len(keys) == 1 else recursive_getvalue(dicts[keys[0]], keys[1:])
    else:
        return None


def recursive_setvalue(keys, value, dicts):
    """Set value of key chain. For exmaple, dicts = {'a':{'b':{'c':1}}}, keys = ['a','b','c'], value = 2
    recursive_setvalue(keys, value, dicts) will set dicts = {'a':{'b':{'c':2}}}
    """
    if len(keys) == 1:
        dicts[keys[0]] = value
    else:
        return recursive_setvalue(keys[1:], value, dicts[keys[0]])


def recursive_getkey(dicts):
    """Return all key chains in the dict. For example, dicts = {'a':{'b':{'c':1},'d':2}},
    recursive_getkey(dicts) will return [['a','b','c'], ['a','d']]
    """
    keys = []
    for key, value in dicts.items():
        if type(value) is dict:
            keys.extend([[key] + _ for _ in recursive_getkey(value)])
        else:
            keys.append([key])
    return keys

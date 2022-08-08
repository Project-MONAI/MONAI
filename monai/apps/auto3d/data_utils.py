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

import os
from copy import deepcopy
from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np
import torch

from monai.bundle.config_parser import ConfigParser
from monai.inferers import SlidingWindowInferer
from monai.transforms.transform import MapTransform

__all__ = [
    "recursive_getvalue",
    "recursive_setvalue",
    "recursive_getkey",
    "datafold_read",
    "distributed_all_gather",
    "AverageMeter",
    "SlidingWindowInfererMemAdapt",
    "LabelMapping",
    "PrintData",
]


def datafold_read(datalist: Union[str, Dict], basedir: str, fold: int = 0, key: str = "training") -> Tuple[List, List]:
    """
    Read a list of data dictionary `datalist`

    Args:
        datalist: the name of a JSON file listing the data, or a dictionary
        basedir: directory of image files
        fold: which fold to use (0..1 if in training set)
        key: usually 'training' , but can try 'validation' or 'testing' to get the list data without labels (used in challenges)

    Returns:
        A tuple of two arrays (training, validation)
    """

    if isinstance(datalist, str):
        json_data = ConfigParser.load_config_file(datalist)
    else:
        json_data = datalist

    dict_data = deepcopy(json_data[key])

    for d in dict_data:
        for k, _ in d.items():
            if isinstance(d[k], list):
                d[k] = [os.path.join(basedir, iv) for iv in d[k]]
            elif isinstance(d[k], str):
                d[k] = os.path.join(basedir, d[k]) if len(d[k]) > 0 else d[k]

    tr = []
    val = []
    for d in dict_data:
        if "fold" in d and d["fold"] == fold:
            val.append(d)
        else:
            tr.append(d)

    return tr, val


def recursive_getvalue(dicts: Dict, keys: Union[str, List]) -> Any:
    """
    Recursively get value through the chain of provided key.

    Args:
        dicts: a provided dictionary
        keys: name of the key or a list showing the key chain to access the value

    Exmaple:
        dicts = {'a':{'b':{'c':1}}}, keys = ['a','b','c']
        recursive_getvalue(dicts, keys) = 1
    """
    if keys[0] in dicts.keys():
        return dicts.get(keys[0]) if len(keys) == 1 else recursive_getvalue(dicts[keys[0]], keys[1:])
    else:
        return None


def recursive_setvalue(keys: Union[str, List], value: Any, dicts: Dict) -> None:
    """
    Recursively set value of key chain.
        dicts: user-provided dictionary

    Args:
        keys: name of the key or a list showing the key chain to access the value
        value: key value
        dicts: user-provided dictionary

    Exmaples:
        dicts = {'a':{'b':{'c':1}}}, keys = ['a','b','c'], value = 2
        recursive_setvalue(keys, value, dicts) will set dicts = {'a':{'b':{'c':2}}}
    """
    if len(keys) == 1:
        dicts[keys[0]] = value
    else:
        recursive_setvalue(keys[1:], value, dicts[keys[0]])


def recursive_getkey(dicts: Dict) -> List:
    """
    Return all key chains in the dict.

    Args:
        dicts: user-provided dictionary

    Returns:
        list of key chains

    Examples:
        dicts = {'a':{'b':{'c':1},'d':2}},
        recursive_getkey(dicts) will return [['a','b','c'], ['a','d']]
    """
    keys = []
    for key, value in dicts.items():
        if type(value) is dict:
            keys.extend([[key] + _ for _ in recursive_getkey(value)])
        else:
            keys.append([key])
    return keys


def distributed_all_gather(tensor_list, out_numpy=False, is_valid: bool = None):
    """ """

    world_size = torch.distributed.get_world_size()

    if is_valid is not None:
        is_valid = torch.tensor(bool(is_valid), dtype=torch.bool, device=tensor_list[0].device)

    tensor_list_out = []
    with torch.no_grad():  # ? do we need it

        if is_valid is not None:
            is_valid_list = [torch.zeros_like(is_valid) for _ in range(world_size)]
            torch.distributed.all_gather(is_valid_list, is_valid)
            is_valid = [x.item() for x in is_valid_list]  # list of bools

        for tensor in tensor_list:
            gather_list = [torch.zeros_like(tensor) for _ in range(world_size)]
            torch.distributed.all_gather(gather_list, tensor)

            if is_valid is not None:
                gather_list = [g for g, v in zip(gather_list, is_valid_list) if v]

            if out_numpy:
                gather_list = [t.cpu().numpy() for t in gather_list]  # convert to numpy

            tensor_list_out.append(gather_list)

    return tensor_list_out


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):

        if not np.isfinite(np.sum(val)) or not np.isfinite(np.sum(n)):
            print("Warning AverageMeter, non-finite input", val, n)
            if not np.isfinite(np.sum(val)):
                val = 0
            if not np.isfinite(np.sum(n)):
                n = 1

        self.val = val
        self.sum += val * n

        self.count += n
        self.avg = np.where(self.count > 0, self.sum / self.count, self.sum)


class SlidingWindowInfererMemAdapt(SlidingWindowInferer):
    def __init__(self, cpu_thresh=512 * 512 * 512, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.cpu_thresh = cpu_thresh

    def __call__(
        self, inputs: torch.Tensor, network: Callable[..., torch.Tensor], *args: Any, **kwargs: Any
    ) -> torch.Tensor:

        vol = inputs.shape[-3] * inputs.shape[-2] * inputs.shape[-1]

        if vol > self.cpu_thresh:  # 512*512*512:
            self.device = torch.device("cpu")  # use stitching on cpu memory if image is too large
        else:
            self.device = None  # same device as input
        return super().__call__(inputs=inputs, network=network, *args, **kwargs)


class LabelMapping(MapTransform):
    """
    Embedd Label according to indices
    """

    def __init__(self, keys: list, class_index: list) -> None:
        super().__init__(keys)
        self.class_index = class_index

    def label_mapping(self, x):
        dtype = x.dtype
        return torch.cat([sum([x == i for i in c]) for c in self.class_index], dim=0).to(dtype=dtype)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = self.label_mapping(d[key])
        return d


class PrintData:
    def __init__(self, desc=None, meta=False):
        self.desc = desc
        self.meta = meta

    def __call__(self, data):
        image = data["image"]
        label = data["label"]
        print(self.desc, "image", image.shape, "type", image.dtype)
        print(self.desc, "label", label.shape, "type", label.dtype)

        if self.meta:
            print(self.desc, "image_meta_dict", data["image_meta_dict"])
            print(self.desc, "label_meta_dict", data["label_meta_dict"])

        return data

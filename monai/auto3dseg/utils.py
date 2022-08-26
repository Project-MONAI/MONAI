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
from numbers import Number
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union, cast

import numpy as np
import torch

from monai.bundle.config_parser import ConfigParser
from monai.bundle.utils import ID_SEP_KEY
from monai.data.meta_tensor import MetaTensor
from monai.transforms import CropForeground, ToCupy
from monai.utils import min_version, optional_import

__all__ = [
    "get_foreground_image",
    "get_foreground_label",
    "get_label_ccp",
    "concat_val_to_np",
    "concat_multikeys_to_dict",
    "datafold_read",
]

measure_np, has_measure = optional_import("skimage.measure", "0.14.2", min_version)
cp, has_cp = optional_import("cupy")
cucim, has_cucim = optional_import("cucim")


def get_foreground_image(image: MetaTensor):
    """
    Get a foreground image by removing all-zero rectangles on the edges of the image
    Note for the developer: update select_fn if the foreground is defined differently.

    Args:
        image: ndarray image to segment.

    Returns:
        ndarray of foreground image by removing all-zero edges.

    Notes:
        the size of the output is smaller than the input.
    """

    copper = CropForeground(select_fn=lambda x: x > 0)
    image_foreground = copper(image)
    return image_foreground


def get_foreground_label(image: MetaTensor, label: MetaTensor) -> MetaTensor:
    """
    Get foreground image pixel values and mask out the non-labeled area.

    Args
        image: ndarray image to segment.
        label: ndarray the image input and annotated with class IDs.

    Returns:
        1D array of foreground image with label > 0
    """

    label_foreground = MetaTensor(image[label > 0])
    return label_foreground


def get_label_ccp(mask_index: MetaTensor, use_gpu: bool = True) -> Tuple[List[Any], int]:
    """
    Find all connected components and their bounding shape. Backend can be cuPy/cuCIM or Numpy
    depending on the hardware.

    Args:
        mask_index: a binary mask.
        use_gpu: a switch to use GPU/CUDA or not. If GPU is unavailable, CPU will be used
            regardless of this setting.

    """

    shape_list = []
    if mask_index.device.type == "cuda" and has_cp and has_cucim and use_gpu:
        mask_cupy = ToCupy()(mask_index.short())
        labeled = cucim.skimage.measure.label(mask_cupy)
        vals = cp.unique(labeled[cp.nonzero(labeled)])

        for ncomp in vals:
            comp_idx = cp.argwhere(labeled == ncomp)
            comp_idx_min = cp.min(comp_idx, axis=0).tolist()
            comp_idx_max = cp.max(comp_idx, axis=0).tolist()
            bbox_shape = [comp_idx_max[i] - comp_idx_min[i] + 1 for i in range(len(comp_idx_max))]
            shape_list.append(bbox_shape)
        ncomponents = len(vals)

    elif has_measure:
        labeled, ncomponents = measure_np.label(mask_index.data.cpu().numpy(), background=-1, return_num=True)
        for ncomp in range(1, ncomponents + 1):
            comp_idx = np.argwhere(labeled == ncomp)
            comp_idx_min = np.min(comp_idx, axis=0).tolist()
            comp_idx_max = np.max(comp_idx, axis=0).tolist()
            bbox_shape = [comp_idx_max[i] - comp_idx_min[i] + 1 for i in range(len(comp_idx_max))]
            shape_list.append(bbox_shape)
    else:
        raise RuntimeError("Cannot find one of the following required dependencies: {cuPy+cuCIM} or {scikit-image}")

    return shape_list, ncomponents


def concat_val_to_np(
    data_list: List[Dict],
    fixed_keys: List[Union[str, int]],
    ragged: Optional[bool] = False,
    allow_missing: Optional[bool] = False,
    **kwargs,
):
    """
    Get the nested value in a list of dictionary that shares the same structure.

    Args:
       data_list: a list of dictionary {key1: {key2: np.ndarray}}.
       fixed_keys: a list of keys that records to path to the value in the dict elements.
       ragged: if True, numbers can be in list of lists or ragged format so concat mode needs change.
       allow_missing: if True, it will return a None if the value cannot be found.

    Returns:
        nd.array of concatenated array.

    """

    np_list: List[Optional[np.ndarray]] = []
    for data in data_list:
        parser = ConfigParser(data)
        for i, key in enumerate(fixed_keys):
            fixed_keys[i] = str(key)

        val: Any
        val = parser.get(ID_SEP_KEY.join(cast(Iterable[str], fixed_keys)))

        if val is None:
            if allow_missing:
                np_list.append(None)
            else:
                raise AttributeError(f"{fixed_keys} is not nested in the dictionary")
        elif isinstance(val, list):
            np_list.append(np.array(val))
        elif isinstance(val, (torch.Tensor, MetaTensor)):
            np_list.append(val.cpu().numpy())
        elif isinstance(val, np.ndarray):
            np_list.append(val)
        elif isinstance(val, Number):
            np_list.append(np.array(val))
        else:
            raise NotImplementedError(f"{val.__class__} concat is not supported.")

    if allow_missing:
        np_list = [x for x in np_list if x is not None]

    if len(np_list) == 0:
        return np.array([0])
    elif ragged:
        return np.concatenate(np_list, **kwargs)  # type: ignore
    else:
        return np.concatenate([np_list], **kwargs)  # type: ignore


def concat_multikeys_to_dict(
    data_list: List[Dict], fixed_keys: List[Union[str, int]], keys: List[str], zero_insert: bool = True, **kwargs
):
    """
    Get the nested value in a list of dictionary that shares the same structure iteratively on all keys.
    It returns a dictionary with keys with the found values in nd.ndarray.

    Args:
        data_list: a list of dictionary {key1: {key2: np.ndarray}}.
        fixed_keys: a list of keys that records to path to the value in the dict elements.
        keys: a list of string keys that will be iterated to generate a dict output.
        zero_insert: insert a zero in the list so that it can find the value in element 0 before getting the keys
        flatten: if True, numbers are flattened before concat.

    Returns:
        a dict with keys - nd.array of concatenated array pair.
    """

    ret_dict = {}
    for key in keys:
        addon: List[Union[str, int]] = [0, key] if zero_insert else [key]
        val = concat_val_to_np(data_list, fixed_keys + addon, **kwargs)
        ret_dict.update({key: val})

    return ret_dict


def datafold_read(datalist: Union[str, Dict], basedir: str, fold: int = 0, key: str = "training") -> Tuple[List, List]:
    """
    Read a list of data dictionary `datalist`

    Args:
        datalist: the name of a JSON file listing the data, or a dictionary.
        basedir: directory of image files.
        fold: which fold to use (0..1 if in training set).
        key: usually 'training' , but can try 'validation' or 'testing' to get the list data without labels (used in challenges).

    Returns:
        A tuple of two arrays (training, validation).
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


def verify_report_format(report: dict, report_format: dict):
    """
    Compares the report and the the report_format that has only keys.

    Args:
        report: dict that has real values.
        report_format: dict that only has keys and list-nested value.
    """
    for k_fmt, v_fmt in report_format.items():
        if k_fmt not in report:
            return False

        v = report[k_fmt]

        if isinstance(v_fmt, list) and isinstance(v, list):
            if len(v_fmt) != 1:
                raise UserWarning("list length in report_format is not 1")
            if len(v_fmt) > 0 and len(v) > 0:
                return verify_report_format(v[0], v_fmt[0])
            else:
                return False

        return True

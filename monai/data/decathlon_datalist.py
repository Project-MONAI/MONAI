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
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union, overload

from monai.config import KeysCollection, PathLike
from monai.data.utils import partition_dataset, select_cross_validation_folds
from monai.utils import ensure_tuple


@overload
def _compute_path(base_dir: PathLike, element: PathLike, check_path: bool = False) -> str:
    ...


@overload
def _compute_path(base_dir: PathLike, element: List[PathLike], check_path: bool = False) -> List[str]:
    ...


def _compute_path(base_dir, element, check_path=False):
    """
    Args:
        base_dir: the base directory of the dataset.
        element: file path(s) to append to directory.
        check_path: if `True`, only compute when the result is an existing path.

    Raises:
        TypeError: When ``element`` contains a non ``str``.
        TypeError: When ``element`` type is not in ``Union[list, str]``.

    """

    def _join_path(base_dir: PathLike, item: PathLike):
        result = os.path.normpath(os.path.join(base_dir, item))
        if check_path and not os.path.exists(result):
            # if not an existing path, don't join with base dir
            return f"{item}"
        return f"{result}"

    if isinstance(element, (str, os.PathLike)):
        return _join_path(base_dir, element)
    if isinstance(element, list):
        for e in element:
            if not isinstance(e, (str, os.PathLike)):
                return element
        return [_join_path(base_dir, e) for e in element]
    return element


def _append_paths(base_dir: PathLike, is_segmentation: bool, items: List[Dict]) -> List[Dict]:
    """
    Args:
        base_dir: the base directory of the dataset.
        is_segmentation: whether the datalist is for segmentation task.
        items: list of data items, each of which is a dict keyed by element names.

    Raises:
        TypeError: When ``items`` contains a non ``dict``.

    """
    for item in items:
        if not isinstance(item, dict):
            raise TypeError(f"Every item in items must be a dict but got {type(item).__name__}.")
        for k, v in item.items():
            if k == "image" or is_segmentation and k == "label":
                item[k] = _compute_path(base_dir, v, check_path=False)
            else:
                # for other items, auto detect whether it's a valid path
                item[k] = _compute_path(base_dir, v, check_path=True)
    return items


def load_decathlon_datalist(
    data_list_file_path: PathLike,
    is_segmentation: bool = True,
    data_list_key: str = "training",
    base_dir: Optional[PathLike] = None,
) -> List[Dict]:
    """Load image/label paths of decathlon challenge from JSON file

    Json file is similar to what you get from http://medicaldecathlon.com/
    Those dataset.json files

    Args:
        data_list_file_path: the path to the json file of datalist.
        is_segmentation: whether the datalist is for segmentation task, default is True.
        data_list_key: the key to get a list of dictionary to be used, default is "training".
        base_dir: the base directory of the dataset, if None, use the datalist directory.

    Raises:
        ValueError: When ``data_list_file_path`` does not point to a file.
        ValueError: When ``data_list_key`` is not specified in the data list file.

    Returns a list of data items, each of which is a dict keyed by element names, for example:

    .. code-block::

        [
            {'image': '/workspace/data/chest_19.nii.gz',  'label': 0},
            {'image': '/workspace/data/chest_31.nii.gz',  'label': 1}
        ]

    """
    data_list_file_path = Path(data_list_file_path)
    if not data_list_file_path.is_file():
        raise ValueError(f"Data list file {data_list_file_path} does not exist.")
    with open(data_list_file_path) as json_file:
        json_data = json.load(json_file)
    if data_list_key not in json_data:
        raise ValueError(f'Data list {data_list_key} not specified in "{data_list_file_path}".')
    expected_data = json_data[data_list_key]
    if data_list_key == "test" and not isinstance(expected_data[0], dict):
        # decathlon datalist may save the test images in a list directly instead of dict
        expected_data = [{"image": i} for i in expected_data]

    if base_dir is None:
        base_dir = data_list_file_path.parent

    return _append_paths(base_dir, is_segmentation, expected_data)


def load_decathlon_properties(data_property_file_path: PathLike, property_keys: Union[Sequence[str], str]) -> Dict:
    """Load the properties from the JSON file contains data property with specified `property_keys`.

    Args:
        data_property_file_path: the path to the JSON file of data properties.
        property_keys: expected keys to load from the JSON file, for example, we have these keys
            in the decathlon challenge:
            `name`, `description`, `reference`, `licence`, `tensorImageSize`,
            `modality`, `labels`, `numTraining`, `numTest`, etc.

    """
    data_property_file_path = Path(data_property_file_path)
    if not data_property_file_path.is_file():
        raise ValueError(f"Data property file {data_property_file_path} does not exist.")
    with open(data_property_file_path) as json_file:
        json_data = json.load(json_file)

    properties = {}
    for key in ensure_tuple(property_keys):
        if key not in json_data:
            raise KeyError(f"key {key} is not in the data property file.")
        properties[key] = json_data[key]
    return properties


def check_missing_files(
    datalist: List[Dict], keys: KeysCollection, root_dir: Optional[PathLike] = None, allow_missing_keys: bool = False
):
    """Checks whether some files in the Decathlon datalist are missing.
    It would be helpful to check missing files before a heavy training run.

    Args:
        datalist: a list of data items, every item is a dictionary.
            usually generated by `load_decathlon_datalist` API.
        keys: expected keys to check in the datalist.
        root_dir: if not None, provides the root dir for the relative file paths in `datalist`.
        allow_missing_keys: whether allow missing keys in the datalist items.
            if False, raise exception if missing. default to False.

    Returns:
        A list of missing filenames.

    """
    missing_files = []
    for item in datalist:
        for k in ensure_tuple(keys):
            if k not in item:
                if not allow_missing_keys:
                    raise ValueError(f"key `{k}` is missing in the datalist item: {item}")
                continue

            for f in ensure_tuple(item[k]):
                if not isinstance(f, (str, os.PathLike)):
                    raise ValueError(f"filepath of key `{k}` must be a string or a list of strings, but got: {f}.")
                f = Path(f)
                if isinstance(root_dir, (str, os.PathLike)):
                    f = Path(root_dir).joinpath(f)
                if not f.exists():
                    missing_files.append(f)

    return missing_files


def create_cross_validation_datalist(
    datalist: List[Dict],
    nfolds: int,
    train_folds: Union[Sequence[int], int],
    val_folds: Union[Sequence[int], int],
    train_key: str = "training",
    val_key: str = "validation",
    filename: Optional[Union[Path, str]] = None,
    shuffle: bool = True,
    seed: int = 0,
    check_missing: bool = False,
    keys: Optional[KeysCollection] = None,
    root_dir: Optional[str] = None,
    allow_missing_keys: bool = False,
    raise_error: bool = True,
):
    """
    Utility to create new Decathlon style datalist based on cross validation partition.

    Args:
        datalist: loaded list of dictionaries for all the items to partition.
        nfolds: number of the kfold split.
        train_folds: indices of folds for training part.
        val_folds: indices of folds for validation part.
        train_key: the key of train part in the new datalist, defaults to "training".
        val_key: the key of validation part in the new datalist, defaults to "validation".
        filename: if not None and ends with ".json", save the new datalist into JSON file.
        shuffle: whether to shuffle the datalist before partition, defaults to `True`.
        seed: if `shuffle` is True, set the random seed, defaults to `0`.
        check_missing: whether to check all the files specified by `keys` are existing.
        keys: if not None and check_missing_files is True, the expected keys to check in the datalist.
        root_dir: if not None, provides the root dir for the relative file paths in `datalist`.
        allow_missing_keys: if check_missing_files is `True`, whether allow missing keys in the datalist items.
            if False, raise exception if missing. default to False.
        raise_error: when found missing files, if `True`, raise exception and stop, if `False`, print warning.

    """
    if check_missing and keys is not None:
        files = check_missing_files(datalist, keys, root_dir, allow_missing_keys)
        if files:
            msg = f"some files of the datalist are missing: {files}"
            if raise_error:
                raise ValueError(msg)
            warnings.warn(msg)

    data = partition_dataset(data=datalist, num_partitions=nfolds, shuffle=shuffle, seed=seed)
    train_list = select_cross_validation_folds(partitions=data, folds=train_folds)
    val_list = select_cross_validation_folds(partitions=data, folds=val_folds)
    ret = {train_key: train_list, val_key: val_list}
    if isinstance(filename, (str, Path)):
        with open(filename, "w") as f:
            json.dump(ret, f, indent=4)

    return ret

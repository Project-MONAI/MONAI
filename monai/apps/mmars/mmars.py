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

"""
Utilities for accessing Nvidia MMARs

See Also:
    - https://docs.nvidia.com/clara/clara-train-sdk/pt/mmar.html
"""

import json
import os
import warnings
from pathlib import Path
from typing import Mapping, Optional, Union

import torch

import monai.networks.nets as monai_nets
from monai.apps.utils import download_and_extract, logger
from monai.config.type_definitions import PathLike
from monai.networks.utils import copy_model_state
from monai.utils.module import optional_import

from .model_desc import MODEL_DESC
from .model_desc import RemoteMMARKeys as Keys

__all__ = ["get_model_spec", "download_mmar", "load_from_mmar"]


def get_model_spec(idx: Union[int, str]):
    """get model specification by `idx`. `idx` could be index of the constant tuple of dict or the actual model ID."""
    if isinstance(idx, int):
        return MODEL_DESC[idx]
    if isinstance(idx, str):
        key = idx.strip().lower()
        for cand in MODEL_DESC:
            if str(cand.get(Keys.ID)).strip().lower() == key:
                return cand
    return idx


def _get_all_ngc_models(pattern, page_index=0, page_size=50):
    url = "https://api.ngc.nvidia.com/v2/search/catalog/resources/MODEL"
    query_dict = {
        "query": "",
        "orderBy": [{"field": "score", "value": "DESC"}],
        "queryFields": ["all", "description", "displayName", "name", "resourceId"],
        "fields": [
            "isPublic",
            "attributes",
            "guestAccess",
            "name",
            "orgName",
            "teamName",
            "displayName",
            "dateModified",
            "labels",
            "description",
        ],
        "page": 0,
    }

    filter = [dict(field="name", value=f"*{pattern}*")]
    query_dict["page"] = page_index
    query_dict["pageSize"] = page_size
    query_dict["filters"] = filter
    query_str = json.dumps(query_dict)
    full_url = f"{url}?q={query_str}"
    requests_get, has_requests = optional_import("requests", name="get")
    if has_requests:
        resp = requests_get(full_url)
        resp.raise_for_status()
    else:
        raise ValueError("NGC API requires requests package.  Please install it.")
    model_list = json.loads(resp.text)
    model_dict = {}
    for result in model_list["results"]:
        for model in result["resources"]:
            current_res_id = model["resourceId"]
            model_dict[current_res_id] = {"name": model["name"]}
            for attribute in model["attributes"]:
                if attribute["key"] == "latestVersionIdStr":
                    model_dict[current_res_id]["latest"] = attribute["value"]
    return model_dict


def _get_ngc_url(model_name: str, version: str, model_prefix=""):
    return f"https://api.ngc.nvidia.com/v2/models/{model_prefix}{model_name}/versions/{version}/zip"


def _get_ngc_doc_url(model_name: str, model_prefix=""):
    return f"https://ngc.nvidia.com/catalog/models/{model_prefix}{model_name}"


def download_mmar(
    item, mmar_dir: Optional[PathLike] = None, progress: bool = True, api: bool = True, version: int = -1
):
    """
    Download and extract Medical Model Archive (MMAR) from Nvidia Clara Train.

    See Also:
        - https://docs.nvidia.com/clara/
        - Nvidia NGC Registry CLI
        - https://docs.nvidia.com/clara/clara-train-sdk/pt/mmar.html

    Args:
        item: the corresponding model item from `MODEL_DESC`.
          Or when api is True, the substring to query NGC's model name field.
        mmar_dir: target directory to store the MMAR, default is `mmars` subfolder under `torch.hub get_dir()`.
        progress: whether to display a progress bar.
        api: whether to query NGC and download via api
        version: which version of MMAR to download.  -1 means the latest from ngc.

    Examples::
        >>> from monai.apps import download_mmar
        >>> download_mmar("clara_pt_prostate_mri_segmentation_1", mmar_dir=".")
        >>> download_mmar("prostate_mri_segmentation", mmar_dir=".", api=True)


    Returns:
        The local directory of the downloaded model.
        If api is True, a list of local directories of downloaded models.
    """
    if not mmar_dir:
        get_dir, has_home = optional_import("torch.hub", name="get_dir")
        if has_home:
            mmar_dir = Path(get_dir()) / "mmars"
        else:
            raise ValueError("mmar_dir=None, but no suitable default directory computed. Upgrade Pytorch to 1.6+ ?")
    mmar_dir = Path(mmar_dir)
    if api:
        model_dict = _get_all_ngc_models(item.get(Keys.NAME, f"{item}") if isinstance(item, Mapping) else f"{item}")
        if len(model_dict) == 0:
            raise ValueError(f"api query returns no item for pattern {item}.  Please change or shorten it.")
        model_dir_list = []
        for k, v in model_dict.items():
            ver = v["latest"] if version == -1 else str(version)
            download_url = _get_ngc_url(k, ver)
            model_dir = mmar_dir / v["name"]
            download_and_extract(
                url=download_url,
                filepath=mmar_dir / f'{v["name"]}_{ver}.zip',
                output_dir=model_dir,
                hash_val=None,
                hash_type="md5",
                file_type="zip",
                has_base=False,
                progress=progress,
            )
            model_dir_list.append(model_dir)
        if not model_dir_list:
            raise ValueError(f"api query download no item for pattern {item}.  Please change or shorten it.")
        return model_dir_list[0]

    if not isinstance(item, Mapping):
        item = get_model_spec(item)
    ver = item.get(Keys.VERSION, 1)
    if version > 0:
        ver = str(version)
    model_fullname = f"{item[Keys.NAME]}_{ver}"
    model_dir = mmar_dir / model_fullname
    model_url = item.get(Keys.URL) or _get_ngc_url(item[Keys.NAME], version=ver, model_prefix="nvidia/med/")
    download_and_extract(
        url=model_url,
        filepath=mmar_dir / f"{model_fullname}.{item[Keys.FILE_TYPE]}",
        output_dir=model_dir,
        hash_val=item[Keys.HASH_VAL],
        hash_type=item[Keys.HASH_TYPE],
        file_type=item[Keys.FILE_TYPE],
        has_base=False,
        progress=progress,
    )
    return model_dir


def load_from_mmar(
    item,
    mmar_dir: Optional[PathLike] = None,
    progress: bool = True,
    version: int = -1,
    map_location=None,
    pretrained=True,
    weights_only=False,
    model_key: str = "model",
    api: bool = True,
    model_file=None,
):
    """
    Download and extract Medical Model Archive (MMAR) model weights from Nvidia Clara Train.

    Args:
        item: the corresponding model item from `MODEL_DESC`.
        mmar_dir: : target directory to store the MMAR, default is mmars subfolder under `torch.hub get_dir()`.
        progress: whether to display a progress bar when downloading the content.
        version: version number of the MMAR. Set it to `-1` to use `item[Keys.VERSION]`.
        map_location: pytorch API parameter for `torch.load` or `torch.jit.load`.
        pretrained: whether to load the pretrained weights after initializing a network module.
        weights_only: whether to load only the weights instead of initializing the network module and assign weights.
        model_key: a key to search in the model file or config file for the model dictionary.
            Currently this function assumes that the model dictionary has
            `{"[name|path]": "test.module", "args": {'kw': 'test'}}`.
        api: whether to query NGC API to get model infomation.
        model_file: the relative path to the model file within an MMAR.

    Examples::
        >>> from monai.apps import load_from_mmar
        >>> unet_model = load_from_mmar("clara_pt_prostate_mri_segmentation_1", mmar_dir=".", map_location="cpu")
        >>> print(unet_model)

    See Also:
        https://docs.nvidia.com/clara/
    """
    if api:
        item = {Keys.NAME: get_model_spec(item)[Keys.NAME] if isinstance(item, int) else f"{item}"}
    if not isinstance(item, Mapping):
        item = get_model_spec(item)
    model_dir = download_mmar(item=item, mmar_dir=mmar_dir, progress=progress, version=version, api=api)
    if model_file is None:
        model_file = os.path.join("models", "model.pt")
    model_file = model_dir / item.get(Keys.MODEL_FILE, model_file)
    logger.info(f'\n*** "{item.get(Keys.NAME)}" available at {model_dir}.')

    # loading with `torch.jit.load`
    if model_file.name.endswith(".ts"):
        if not pretrained:
            warnings.warn("Loading a ScriptModule, 'pretrained' option ignored.")
        if weights_only:
            warnings.warn("Loading a ScriptModule, 'weights_only' option ignored.")
        return torch.jit.load(model_file, map_location=map_location)

    # loading with `torch.load`
    model_dict = torch.load(model_file, map_location=map_location)
    if weights_only:
        return model_dict.get(model_key, model_dict)  # model_dict[model_key] or model_dict directly

    # 1. search `model_dict['train_config]` for model config spec.
    model_config = _get_val(dict(model_dict).get("train_conf", {}), key=model_key, default={})
    if not model_config or not isinstance(model_config, Mapping):
        # 2. search json CONFIG_FILE for model config spec.
        json_path = model_dir / item.get(Keys.CONFIG_FILE, os.path.join("config", "config_train.json"))
        with open(json_path) as f:
            conf_dict = json.load(f)
        conf_dict = dict(conf_dict)
        model_config = _get_val(conf_dict, key=model_key, default={})
    if not model_config:
        # 3. search `model_dict` for model config spec.
        model_config = _get_val(dict(model_dict), key=model_key, default={})

    if not (model_config and isinstance(model_config, Mapping)):
        raise ValueError(
            f"Could not load model config dictionary from config: {item.get(Keys.CONFIG_FILE)}, "
            f"or from model file: {item.get(Keys.MODEL_FILE)}."
        )

    # parse `model_config` for model class and model parameters
    if model_config.get("name"):  # model config section is a "name"
        model_name = model_config["name"]
        model_cls = monai_nets.__dict__[model_name]
    elif model_config.get("path"):  # model config section is a "path"
        # https://docs.nvidia.com/clara/clara-train-sdk/pt/byom.html
        model_module, model_name = model_config.get("path", ".").rsplit(".", 1)
        model_cls, has_cls = optional_import(module=model_module, name=model_name)
        if not has_cls:
            raise ValueError(
                f"Could not load MMAR model config {model_config.get('path', '')}, "
                f"Please make sure MMAR's sub-folders in '{model_dir}' is on the PYTHONPATH."
                "See also: https://docs.nvidia.com/clara/clara-train-sdk/pt/byom.html"
            )
    else:
        raise ValueError(f"Could not load model config {model_config}.")

    logger.info(f"*** Model: {model_cls}")
    model_kwargs = model_config.get("args", None)
    if model_kwargs:
        model_inst = model_cls(**model_kwargs)
        logger.info(f"*** Model params: {model_kwargs}")
    else:
        model_inst = model_cls()
    if pretrained:
        _, changed, unchanged = copy_model_state(model_inst, model_dict.get(model_key, model_dict), inplace=True)
        if not (changed and not unchanged):  # not all model_inst variables are changed
            logger.warning(f"*** Loading model state -- unchanged: {len(unchanged)}, changed: {len(changed)}.")
    logger.info("\n---")
    doc_url = item.get(Keys.DOC) or _get_ngc_doc_url(item[Keys.NAME], model_prefix="nvidia:med:")
    logger.info(f"For more information, please visit {doc_url}\n")
    return model_inst


def _get_val(input_dict: Mapping, key="model", default=None):
    """
    Search for the item with `key` in `config_dict`.
    Returns: the first occurrence of `key` in a breadth first search.
    """
    if key in input_dict:
        return input_dict[key]
    for sub_dict in input_dict:
        val = input_dict[sub_dict]
        if isinstance(val, Mapping):
            found_val = _get_val(val, key=key, default=None)
            if found_val is not None:
                return found_val
    return default

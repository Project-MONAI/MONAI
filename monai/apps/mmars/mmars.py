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
Deprecated utilities for accessing Nvidia MMARs. Use MONAI Bundles instead.

See Also:
    - https://docs.nvidia.com/clara/clara-train-sdk/pt/mmar.html
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from monai.apps.utils import download_and_extract
from monai.config.type_definitions import PathLike
from monai.utils.deprecate_utils import DeprecatedError, deprecated
from monai.utils.module import optional_import

from .model_desc import MODEL_DESC
from .model_desc import RemoteMMARKeys as Keys

__all__ = ["get_model_spec", "download_mmar", "load_from_mmar"]


def _get_model_spec(idx: int | str) -> dict | Any:
    """
    Look up a model specification without emitting a deprecation warning.

    Args:
        idx: An index into ``MODEL_DESC`` or a model ID. Negative indices count from the end.
            Model IDs ignore case and leading or trailing whitespace.

    Returns:
        The matching model specification. Unknown IDs and inputs other than integers or strings
        are returned unchanged.

    Raises:
        IndexError: If an integer index is outside the range of ``MODEL_DESC``.
    """
    if isinstance(idx, int):
        return MODEL_DESC[idx]
    if isinstance(idx, str):
        key = idx.strip().lower()
        for cand in MODEL_DESC:
            if str(cand.get(Keys.ID)).strip().lower() == key:
                return cand
    return idx


@deprecated(
    since="1.7.0",
    removed="1.9.0",
    msg_suffix="NVIDIA Clara MMAR support is deprecated. Please use MONAI Bundles instead.",
)
def get_model_spec(idx: int | str) -> dict | Any:
    """
    Get a model specification by index or model ID.

    .. deprecated:: 1.7.0
        NVIDIA Clara MMAR support will be removed in version 1.9.0. Use MONAI Bundles instead.
    """
    return _get_model_spec(idx)


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


def _get_ngc_url(model_name: str, version: str, model_prefix: str = "") -> str:
    return f"https://api.ngc.nvidia.com/v2/models/{model_prefix}{model_name}/versions/{version}/zip"


@deprecated(
    since="1.7.0",
    removed="1.9.0",
    msg_suffix="NVIDIA Clara MMAR support is deprecated. Please use MONAI Bundles instead.",
)
def download_mmar(
    item: str | Mapping, mmar_dir: PathLike | None = None, progress: bool = True, api: bool = True, version: int = -1
) -> Path:
    """
    Download and extract Medical Model Archive (MMAR) from Nvidia Clara Train.

    .. deprecated:: 1.7.0
        NVIDIA Clara MMAR support will be removed in version 1.9.0. Use MONAI Bundles instead.

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
    _mmar_dir = Path(mmar_dir)
    model_dir: Path
    if api:
        model_dict = _get_all_ngc_models(item.get(Keys.NAME, f"{item}") if isinstance(item, Mapping) else f"{item}")
        if len(model_dict) == 0:
            raise ValueError(f"api query returns no item for pattern {item}.  Please change or shorten it.")
        model_dir_list: list[Path] = []
        for k, v in model_dict.items():
            ver = v["latest"] if version == -1 else str(version)
            download_url = _get_ngc_url(k, ver)
            model_dir = _mmar_dir / v["name"]
            download_and_extract(
                url=download_url,
                filepath=_mmar_dir / f'{v["name"]}_{ver}.zip',
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
        item = _get_model_spec(item)
    ver = item.get(Keys.VERSION, 1)
    if version > 0:
        ver = str(version)
    model_fullname = f"{item[Keys.NAME]}_{ver}"
    model_dir = _mmar_dir / model_fullname
    model_url = item.get(Keys.URL) or _get_ngc_url(item[Keys.NAME], version=ver, model_prefix="nvidia/med/")
    download_and_extract(
        url=model_url,
        filepath=_mmar_dir / f"{model_fullname}.{item[Keys.FILE_TYPE]}",
        output_dir=model_dir,
        hash_val=item[Keys.HASH_VAL],
        hash_type=item[Keys.HASH_TYPE],
        file_type=item[Keys.FILE_TYPE],
        has_base=False,
        progress=progress,
    )
    return model_dir


@deprecated(
    since="1.7.0",
    removed="1.9.0",
    msg_suffix="NVIDIA Clara MMAR support is deprecated. Please use MONAI Bundles instead.",
)
def load_from_mmar(
    item: Mapping | str | int,
    mmar_dir: PathLike | None = None,
    progress: bool = True,
    version: int = -1,
    map_location: Any | None = None,
    pretrained: bool = True,
    weights_only: bool = False,
    model_key: str = "model",
    api: bool = True,
    model_file: PathLike | None = None,
) -> Any:
    """
    Disabled loader for NVIDIA Clara Medical Model Archive (MMAR) models.

    .. deprecated:: 1.7.0
        Loading MMAR models is disabled for security reasons. This function will be removed
        in version 1.9.0. Use MONAI Bundles instead.

    Every call raises :class:`monai.utils.DeprecatedError` before downloading or
    loading a model. The arguments are kept for compatibility and are ignored.
    Their former meanings are listed below.

    Args:
        item: the corresponding model item from `MODEL_DESC`.
        mmar_dir: target directory to store the MMAR, default is mmars subfolder under `torch.hub get_dir()`.
        progress: whether to display a progress bar when downloading the content.
        version: version number of the MMAR. Set it to `-1` to use `item[Keys.VERSION]`.
        map_location: pytorch API parameter for `torch.load` or `torch.jit.load`.
        pretrained: whether to load the pretrained weights after initializing a network module.
        weights_only: whether to load only the weights instead of initializing the network module and assign weights.
        model_key: a key to search in the model file or config file for the model dictionary.
            The model dictionary was expected to have
            `{"[name|path]": "test.module", "args": {'kw': 'test'}}`.
        api: whether to query NGC API to get model information.
        model_file: the relative path to the model file within an MMAR.

    Raises:
        DeprecatedError: always, because MMAR loading is disabled for security reasons.
    """
    raise DeprecatedError("load_from_mmar is disabled for security reasons. Please use MONAI Bundles instead.")

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

import os
import warnings
from typing import Mapping

import torch

import monai.networks.nets as monai_nets
from monai.apps.utils import download_and_extract

__all__ = ["download_mmar", "load_from_mmar", "MODEL_DESC"]


MODEL_DESC = (
    {
        "id": "clara_pt_prostate_mri_segmentation_1",
        "name": "clara_pt_prostate_mri_segmentation",
        "url": "https://api.ngc.nvidia.com/v2/models/nvidia/med/clara_pt_prostate_mri_segmentation/versions/1/zip",
        "doc": "https://ngc.nvidia.com/catalog/models/nvidia:med:clara_pt_prostate_mri_segmentation",
        "file_type": "zip",
        "hash_type": "md5",
        "hash_val": None,
        "model_file": os.path.join("models", "model.pt"),
    },
    {
        "id": "clara_pt_covid19_ct_lesion_segmentation_1",
        "name": "clara_pt_covid19_ct_lesion_segmentation",
        "url": "https://api.ngc.nvidia.com/v2/models/nvidia/med/clara_pt_covid19_ct_lesion_segmentation/versions/1/zip",
        "doc": "https://ngc.nvidia.com/catalog/models/nvidia:med:clara_pt_covid19_ct_lesion_segmentation",
        "file_type": "zip",
        "hash_type": "md5",
        "hash_val": None,
        "model_file": os.path.join("models", "model.pt"),
    },
)
"""
Collection of the remote MMAR descriptors
"""


def _get_model_spec(idx):
    """get model specification by `idx`. `idx` could be index of the constant tuple of dict or the actual model ID."""
    if isinstance(idx, int):
        return MODEL_DESC[idx]
    if isinstance(idx, str):
        key = idx.strip().lower()
        for cand in MODEL_DESC:
            if cand["id"].strip().lower() == key:
                return cand
    print(f"Available specs are: {MODEL_DESC}.")
    raise ValueError(f"Unknown MODEL_DESC request: {idx}")


def download_mmar(item, mmar_dir=None, progress: bool = True):
    """
    Download and extract Medical Model Archive (MMAR) from Nvidia Clara Train.

    See Also:
        https://docs.nvidia.com/clara/

    Args:
        item: the corresponding model item from `MODEL_DESC`.
        mmar_dir: target directory to store the MMAR, default is mmars subfolder under `torch.hub get_dir()`.
        progress: whether to display a progress bar.

    Returns:
        The local directory of the downloaded model.
    """
    if not isinstance(item, Mapping):
        item = _get_model_spec(item)
    if not mmar_dir:
        mmar_dir = os.path.join(torch.hub.get_dir(), "mmars")
    model_dir = os.path.join(mmar_dir, item["id"])
    download_and_extract(
        url=item["url"],
        filepath=os.path.join(mmar_dir, f"{item['id']}.{item['file_type']}"),
        output_dir=model_dir,
        hash_val=item["hash_val"],
        hash_type=item["hash_type"],
        file_type=item["file_type"],
        has_base=False,
        progress=progress,
    )
    print(f"*** \"{item['id']}\" available at {model_dir}.\n")
    return model_dir


def load_from_mmar(item, mmar_dir=None, progress: bool = True, map_location=None, pretrained=True, weights_only=False):
    """
    Download and extract Medical Model Archive (MMAR) model weights from Nvidia Clara Train.

    Args:
        item: the corresponding model item from `MODEL_DESC`.
        mmar_dir: : target directory to store the MMAR, default is mmars subfolder under `torch.hub get_dir()`.
        progress: whether to display a progress bar when downloading the content.
        map_location: pytorch API parameter for `torch.load` or `torch.jit.load`.
        pretrained: whether to load the pretrained weights after initializing a network module.
        weights_only: whether to load only the weights instead of initializing the network module and assign weights.

    See Also:
        https://docs.nvidia.com/clara/
    """
    if not isinstance(item, Mapping):
        item = _get_model_spec(item)
    model_dir = download_mmar(item=item, mmar_dir=mmar_dir, progress=progress)
    model_file = os.path.join(model_dir, item["model_file"])

    # loading with `torch.jit.load`
    if f"{model_file}".endswith(".ts"):
        if not pretrained:
            warnings.warn("Loading a ScriptModule, 'pretrained' option ignored.")
        if weights_only:
            warnings.warn("Loading a ScriptModule, 'weights_only' option ignored.")
        return torch.jit.load(model_file, map_location=map_location)

    # loading with `torch.load`
    model_dict = torch.load(model_file, map_location=map_location)
    if weights_only:
        return model_dict["model"]

    # TODO: search for the module based on model name?
    # TODO：support [model][path]?
    model_config = model_dict["train_conf"]["train"]["model"]
    model_name = model_config["name"]
    model_kwargs = model_config["args"]
    model_cls = monai_nets.__dict__[model_name]
    model_inst = model_cls(**model_kwargs)
    print(f"{model_cls}({model_kwargs})")
    if pretrained:
        model_inst.load_state_dict(model_dict["model"])
    print("\n---")
    print(f"For more information, please visit {item['doc']}\n")
    return model_inst

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
import zipfile
from typing import Any

from monai.config.deviceconfig import get_config_values
from monai.utils import optional_import

yaml, _ = optional_import("yaml")

__all__ = ["ID_REF_KEY", "ID_SEP_KEY", "EXPR_KEY", "MACRO_KEY"]

ID_REF_KEY = "@"  # start of a reference to a ConfigItem
ID_SEP_KEY = "#"  # separator for the ID of a ConfigItem
EXPR_KEY = "$"  # start of a ConfigExpression
MACRO_KEY = "%"  # start of a macro of a config

_conf_values = get_config_values()

DEFAULT_METADATA = {
    "version": "0.0.1",
    "changelog": {"0.0.1": "Initial version"},
    "monai_version": _conf_values["MONAI"],
    "pytorch_version": _conf_values["Pytorch"],
    "numpy_version": _conf_values["Numpy"],
    "optional_packages_version": {},
    "task": "Describe what the network predicts",
    "description": "A longer description of what the network does, use context, inputs, outputs, etc.",
    "authors": "Your Name Here",
    "copyright": "Copyright (c) Your Name Here",
    "network_data_format": {"inputs": {}, "outputs": {}},
}

DEFAULT_INFERENCE = {
    "imports": ["$import glob"],
    "device": "$torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')",
    "ckpt_path": "$@bundle_root + '/models/model.pt'",
    "dataset_dir": "/workspace/data",
    "datalist": "$list(sorted(glob.glob(@dataset_dir + '/*.jpeg')))",
    "network_def": {"_target_": "???", "spatial_dims": 2},
    "network": "$@network_def.to(@device)",
    "preprocessing": {
        "_target_": "Compose",
        "transforms": [
            {"_target_": "LoadImaged", "keys": "image"},
            {"_target_": "EnsureChannelFirstd", "keys": "image"},
            {"_target_": "ScaleIntensityd", "keys": "image"},
            {"_target_": "EnsureTyped", "keys": "image", "device": "@device"},
        ],
    },
    "dataset": {"_target_": "Dataset", "data": "$[{'image': i} for i in @datalist]", "transform": "@preprocessing"},
    "dataloader": {
        "_target_": "DataLoader",
        "dataset": "@dataset",
        "batch_size": 1,
        "shuffle": False,
        "num_workers": 0,
    },
    "inferer": {"_target_": "SimpleInferer"},
    "postprocessing": {
        "_target_": "Compose",
        "transforms": [
            {"_target_": "Activationsd", "keys": "pred", "softmax": True},
            {"_target_": "AsDiscreted", "keys": "pred", "argmax": True},
        ],
    },
    "handlers": [
        {
            "_target_": "CheckpointLoader",
            "_disabled_": "$not os.path.exists(@ckpt_path)",
            "load_path": "@ckpt_path",
            "load_dict": {"model": "@network"},
        }
    ],
    "evaluator": {
        "_target_": "SupervisedEvaluator",
        "device": "@device",
        "val_data_loader": "@dataloader",
        "network": "@network",
        "inferer": "@inferer",
        "postprocessing": "@postprocessing",
        "val_handlers": "@handlers",
    },
    "evaluating": ["$@evaluator.run()"],
}

DEFAULT_HANDLERS_ID = {
    "trainer": {"id": "train#trainer", "handlers": "train#handlers"},
    "validator": {"id": "validate#evaluator", "handlers": "validate#handlers"},
    "evaluator": {"id": "evaluator", "handlers": "handlers"},
}

DEFAULT_MLFLOW_SETTINGS = {
    "handlers_id": DEFAULT_HANDLERS_ID,
    "configs": {
        # MLFlowHandler config for the trainer
        "trainer": {
            "_target_": "MLFlowHandler",
            "tracking_uri": "$@output_dir + '/mlflow'",
            "iteration_log": True,
            "epoch_log": True,
            "tag_name": "train_loss",
            "output_transform": "$monai.handlers.from_engine(['loss'], first=True)",
        },
        # MLFlowHandler config for the validator
        "validator": {"_target_": "MLFlowHandler", "tracking_uri": "$@output_dir + '/mlflow'", "iteration_log": False},
        # MLFlowHandler config for the evaluator
        "evaluator": {"_target_": "MLFlowHandler", "tracking_uri": "$@output_dir + '/mlflow'", "iteration_log": False},
    },
}


def load_bundle_config(bundle_path: str, *config_names, **load_kw_args) -> Any:
    """
    Load the metadata and nominated configuration files from a MONAI bundle without loading the network itself.

    This function will load the information from the bundle, which can be a directory or a zip file containing a
    directory or a Torchscript bundle, and return the parser object with the information. This saves having to load
    the model if only the information is wanted, and can work on any sort of bundle format.

    Args:
        bundle_path: path to the bundle directory or zip file
        config_names: names of configuration files with extensions to load, should not be full paths but just name+ext
        load_kw_args: keyword arguments to pass to the ConfigParser object when loading

    Returns:
        ConfigParser object containing the parsed information
    """

    from monai.bundle.config_parser import ConfigParser  # avoids circular import

    parser = ConfigParser()

    if not os.path.exists(bundle_path):
        raise ValueError(f"Cannot find bundle file/directory '{bundle_path}'")

    # bundle is a directory, read files directly
    if os.path.isdir(bundle_path):
        conf_data = []
        parser.read_meta(f=os.path.join(bundle_path, "configs", "metadata.json"), **load_kw_args)

        for cname in config_names:
            cpath = os.path.join(bundle_path, "configs", cname)
            if not os.path.exists(cpath):
                raise ValueError(f"Cannot find config file '{cpath}'")

            conf_data.append(cpath)

        parser.read_config(f=conf_data, **load_kw_args)
    else:
        # bundle is a zip file which is either a zipped directory or a Torchscript archive

        name, _ = os.path.splitext(os.path.basename(bundle_path))

        archive = zipfile.ZipFile(bundle_path, "r")

        all_files = archive.namelist()

        zip_meta_name = f"{name}/configs/metadata.json"

        if zip_meta_name in all_files:
            prefix = f"{name}/configs/"  # zipped directory location for files
        else:
            zip_meta_name = f"{name}/extra/metadata.json"
            prefix = f"{name}/extra/"  # Torchscript location for files

        meta_json = json.loads(archive.read(zip_meta_name))
        parser.read_meta(f=meta_json)

        for cname in config_names:
            full_cname = prefix + cname
            if full_cname not in all_files:
                raise ValueError(f"Cannot find config file '{full_cname}'")

            ardata = archive.read(full_cname)

            if full_cname.lower().endswith("json"):
                cdata = json.loads(ardata, **load_kw_args)
            elif full_cname.lower().endswith(("yaml", "yml")):
                cdata = yaml.safe_load(ardata, **load_kw_args)

            parser.read_config(f=cdata)

    return parser

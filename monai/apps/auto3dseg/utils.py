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

from __future__ import annotations

import os
from typing import Any

from monai.apps.auto3dseg.bundle_gen import BundleAlgo
from monai.auto3dseg import algo_from_pickle, algo_to_pickle
from monai.utils.module import optional_import

health_azure, has_health_azure = optional_import("health_azure")

AZUREML_CONFIG_KEY = "azureml_config"


def import_bundle_algo_history(
    output_folder: str = ".", template_path: str | None = None, only_trained: bool = True
) -> list:
    """
    import the history of the bundleAlgo object with their names/identifiers

    Args:
        output_folder: the root path of the algorithms templates.
        template_path: the algorithm_template. It must contain algo.py in the follow path:
            ``{algorithm_templates_dir}/{network}/scripts/algo.py``.
        only_trained: only read the algo history if the algo is trained.
    """

    history = []

    for name in sorted(os.listdir(output_folder)):
        write_path = os.path.join(output_folder, name)

        if not os.path.isdir(write_path):
            continue

        obj_filename = os.path.join(write_path, "algo_object.pkl")
        if not os.path.isfile(obj_filename):  # saved mode pkl
            continue

        algo, algo_meta_data = algo_from_pickle(obj_filename, template_path=template_path)

        if isinstance(algo, BundleAlgo):  # algo's template path needs override
            algo.template_path = algo_meta_data["template_path"]

        if only_trained:
            if "best_metrics" in algo_meta_data:
                history.append({name: algo})
        else:
            history.append({name: algo})

    return history


def export_bundle_algo_history(history: list[dict[str, BundleAlgo]]) -> None:
    """
    Save all the BundleAlgo in the history to algo_object.pkl in each individual folder

    Args:
        history: a List of Bundle. Typically, the history can be obtained from BundleGen get_history method
    """
    for task in history:
        for _, algo in task.items():
            algo_to_pickle(algo, template_path=algo.template_path)


def submit_auto3dseg_module_to_azureml_if_needed(azure_cfg: dict[str, Any]) -> health_azure.AzureRunInfo:
    """
    Submit Auto3dSeg modules to run as AzureML jobs if the user has requested it.

    Args:
        azure_cfg: Dictionary containing arguments to be used for AzureML job submission.
    """
    azureml_args = {
        "workspace_config_file": "azureml_configs/azureml_config.json",
        "docker_base_image": "mcr.microsoft.com/azureml/openmpi3.1.2-cuda10.2-cudnn8-ubuntu18.04",
        "snapshot_root_directory": os.getcwd(),
        "conda_environment_file": "environment-azureml.yml",
        "entry_script": "-m monai.apps.auto3dseg",
        "strictly_aml_v1": False,
    }
    azureml_args.update(azure_cfg)

    needed_keys = {"compute_cluster_name", "default_datastore"}
    missing_keys = needed_keys.difference(azureml_args.keys())
    if len(missing_keys) > 0:
        raise ValueError(f"Missing keys in azureml_args: {missing_keys}")

    run_info = health_azure.submit_to_azure_if_needed(**azureml_args)

    return run_info

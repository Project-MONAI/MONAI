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

import subprocess
from pathlib import Path

import yaml
from pyhocon import ConfigFactory
from pyhocon.converter import HOCONConverter


def prepare_config(clients, experiment, root_dir, script_dir, nvflare_exec):
    """
    Prepare configuration files for nnUNet dataset preparation using NVFlare.

    Parameters
    ----------
    clients : dict
        Dictionary containing client-specific configurations. Each key is a client ID and the value is a dictionary
        with the following keys:
        - "data_dir": str, path to the client's data directory.
        - "patient_id_in_file_identifier": str, identifier for patient ID in file.
        - "modality_dict": dict, dictionary mapping modalities.
        - "dataset_format": str, format of the dataset.
        - "nnunet_root_folder": str, path to the nnUNet root folder.
        - "client_name": str, name of the client.
        - "subfolder_suffix": str, optional, suffix for subfolders.
    experiment : dict
        Dictionary containing experiment-specific configurations with the following keys:
        - "dataset_name_or_id": str, name or ID of the dataset.
        - "experiment_name": str, name of the experiment.
        - "tracking_uri": str, URI for tracking.
        - "mlflow_token": str, optional, token for MLflow.
    root_dir : str
        Root directory where the configuration files will be generated.
    script_dir : str
        Directory containing the scripts.
    nvflare_exec : str
        Path to the NVFlare executable.

    Returns
    -------
    None
    """
    task_name = "prepare"
    Path(root_dir).joinpath(task_name).mkdir(parents=True, exist_ok=True)

    info = {"description": "Prepare nnUNet Dataset", "client_category": "Executor", "controller_type": "server"}

    meta = {
        "name": f"{task_name}_nnUNet",
        "resource_spec": {},
        "deploy_map": {f"{task_name}-server": ["server"]},
        "min_clients": 1,
        "mandatory_clients": list(clients.keys()),
    }
    for client_id in clients:
        meta["deploy_map"][f"{task_name}-client-{client_id}"] = [client_id]

    with open(Path(root_dir).joinpath(task_name).joinpath("info.conf"), "w") as f:
        f.write("{\n")
        f.write(HOCONConverter.to_hocon(ConfigFactory.from_dict(info)))
        f.write("\n}")

    with open(Path(root_dir).joinpath(task_name).joinpath("meta.conf"), "w") as f:
        f.write("{\n")
        f.write(HOCONConverter.to_hocon(ConfigFactory.from_dict(meta)))
        f.write("\n}")

    server = {
        "format_version": 2,
        "server": {"heart_beat_timeout": 600},
        "task_data_filters": [],
        "task_result_filters": [],
        "components": [
            {"id": "nnunet_processor", "path": "monai.nvflare.response_processor.nnUNetPrepareProcessor", "args": {}},
            {"id": "json_generator", "path": "monai.nvflare.json_generator.PrepareJsonGenerator", "args": {}},
        ],
        "workflows": [
            {
                "id": "broadcast_and_process",
                "name": "BroadcastAndProcess",
                "args": {
                    "processor": "nnunet_processor",
                    "min_responses_required": 0,
                    "wait_time_after_min_received": 10,
                    "task_name": task_name,
                    "timeout": 6000,
                },
            }
        ],
    }
    Path(root_dir).joinpath(task_name).joinpath(f"{task_name}-server").mkdir(parents=True, exist_ok=True)
    with open(Path(root_dir).joinpath(task_name).joinpath(f"{task_name}-server", "config_fed_server.conf"), "w") as f:
        f.write("{\n")
        f.write(HOCONConverter.to_hocon(ConfigFactory.from_dict(server)))
        f.write("\n}")

    for client_id in clients:
        client = {
            "format_version": 2,
            "task_result_filters": [],
            "task_data_filters": [],
            "components": [],
            "executors": [
                {
                    "tasks": [task_name],
                    "executor": {
                        "path": "monai.nvflare.nnunet_executor.nnUNetExecutor",
                        "args": {
                            "data_dir": clients[client_id]["data_dir"],
                            "patient_id_in_file_identifier": clients[client_id]["patient_id_in_file_identifier"],
                            "modality_dict": clients[client_id]["modality_dict"],
                            "dataset_format": clients[client_id]["dataset_format"],
                            "nnunet_root_folder": clients[client_id]["nnunet_root_folder"],
                            "nnunet_config": {
                                "dataset_name_or_id": experiment["dataset_name_or_id"],
                                "experiment_name": experiment["experiment_name"],
                            },
                            "client_name": clients[client_id]["client_name"],
                            "tracking_uri": experiment["tracking_uri"],
                        },
                    },
                }
            ],
        }

        if "modality_list" in experiment:
            client["executors"][0]["executor"]["args"]["modality_list"] = experiment["modality_list"]

        if "subfolder_suffix" in clients[client_id]:
            client["executors"][0]["executor"]["args"]["subfolder_suffix"] = clients[client_id]["subfolder_suffix"]
        if "mlflow_token" in experiment:
            client["executors"][0]["executor"]["args"]["mlflow_token"] = experiment["mlflow_token"]

        if "nnunet_plans" in experiment:
            client["executors"][0]["executor"]["args"]["nnunet_config"]["nnunet_plans"] = experiment["nnunet_plans"]

        if "nnunet_trainer" in experiment:
            client["executors"][0]["executor"]["args"]["nnunet_config"]["nnunet_trainer"] = experiment["nnunet_trainer"]

        Path(root_dir).joinpath(task_name).joinpath(f"{task_name}-client-{client_id}").mkdir(
            parents=True, exist_ok=True
        )
        with open(
            Path(root_dir).joinpath(task_name).joinpath(f"{task_name}-client-{client_id}", "config_fed_client.conf"),
            "w",
        ) as f:
            f.write("{\n")
            f.write(HOCONConverter.to_hocon(ConfigFactory.from_dict(client)))
            f.write("\n}")

    subprocess.run(
        [
            nvflare_exec,
            "job",
            "create",
            "-j",
            Path(root_dir).joinpath("jobs", task_name),
            "-w",
            Path(root_dir).joinpath(task_name),
            "-sd",
            script_dir,
            "--force",
        ]
    )


def check_client_packages_config(clients, experiment, root_dir, script_dir, nvflare_exec):
    """
    Generate job configuration files for checking client packages in an NVFlare experiment.

    Parameters
    ----------
    clients : dict
        A dictionary where keys are client IDs and values are client details.
    experiment : str
        The name of the experiment.
    root_dir : str
        The root directory where the configuration files will be generated.
    script_dir : str
        The directory containing the necessary scripts for NVFlare.
    nvflare_exec : str
        The NVFlare executable path.

    Returns
    -------
    None
    """
    task_name = "check_client_packages"
    Path(root_dir).joinpath(task_name).mkdir(parents=True, exist_ok=True)

    info = {
        "description": "Check Python Packages and Report",
        "client_category": "Executor",
        "controller_type": "server",
    }

    meta = {
        "name": f"{task_name}",
        "resource_spec": {},
        "deploy_map": {f"{task_name}-server": ["server"]},
        "min_clients": 1,
        "mandatory_clients": list(clients.keys()),
    }
    for client_id in clients:
        meta["deploy_map"][f"{task_name}-client-{client_id}"] = [client_id]

    with open(Path(root_dir).joinpath(task_name).joinpath("info.conf"), "w") as f:
        f.write("{\n")
        f.write(HOCONConverter.to_hocon(ConfigFactory.from_dict(info)))
        f.write("\n}")

    with open(Path(root_dir).joinpath(task_name).joinpath("meta.conf"), "w") as f:
        f.write("{\n")
        f.write(HOCONConverter.to_hocon(ConfigFactory.from_dict(meta)))
        f.write("\n}")

    server = {
        "format_version": 2,
        "server": {"heart_beat_timeout": 600},
        "task_data_filters": [],
        "task_result_filters": [],
        "components": [
            {
                "id": "nnunet_processor",
                "path": "monai.nvflare.response_processor.nnUNetPackageReportProcessor",
                "args": {},
            },
            {
                "id": "json_generator",
                "path": "monai.nvflare.json_generator.nnUNetPackageReportJsonGenerator",
                "args": {},
            },
        ],
        "workflows": [
            {
                "id": "broadcast_and_process",
                "name": "BroadcastAndProcess",
                "args": {
                    "processor": "nnunet_processor",
                    "min_responses_required": 0,
                    "wait_time_after_min_received": 10,
                    "task_name": task_name,
                    "timeout": 6000,
                },
            }
        ],
    }
    Path(root_dir).joinpath(task_name).joinpath(f"{task_name}-server").mkdir(parents=True, exist_ok=True)
    with open(Path(root_dir).joinpath(task_name).joinpath(f"{task_name}-server", "config_fed_server.conf"), "w") as f:
        f.write("{\n")
        f.write(HOCONConverter.to_hocon(ConfigFactory.from_dict(server)))
        f.write("\n}")

    for client_id in clients:
        client = {
            "format_version": 2,
            "task_result_filters": [],
            "task_data_filters": [],
            "components": [],
            "executors": [
                {"tasks": [task_name], "executor": {"path": "monai.nvflare.nnunet_executor.nnUNetExecutor", "args": {}}}
            ],
        }

        Path(root_dir).joinpath(task_name).joinpath(f"{task_name}-client-{client_id}").mkdir(
            parents=True, exist_ok=True
        )
        with open(
            Path(root_dir).joinpath(task_name).joinpath(f"{task_name}-client-{client_id}", "config_fed_client.conf"),
            "w",
        ) as f:
            f.write("{\n")
            f.write(HOCONConverter.to_hocon(ConfigFactory.from_dict(client)))
            f.write("\n}")

    subprocess.run(
        [
            nvflare_exec,
            "job",
            "create",
            "-j",
            Path(root_dir).joinpath("jobs", task_name),
            "-w",
            Path(root_dir).joinpath(task_name),
            "-sd",
            script_dir,
            "--force",
        ]
    )


def plan_and_preprocess_config(clients, experiment, root_dir, script_dir, nvflare_exec):
    """
    Generates and writes configuration files for the plan and preprocess task in the nnUNet experiment.

    Parameters
    ----------
    clients : dict
        A dictionary containing client-specific configurations. Each key is a client ID, and the value is
        another dictionary with client-specific settings.
    experiment : dict
        A dictionary containing experiment-specific configurations such as dataset name, experiment name,
        tracking URI, and optional nnUNet plans and trainer.
    root_dir : str
        The root directory where the configuration files will be generated.
    script_dir : str
        The directory containing the scripts to be used in the NVFlare job.
    nvflare_exec : str
        The path to the NVFlare executable.

    Returns
    -------
    None
    """
    task_name = "plan_and_preprocess"
    Path(root_dir).joinpath(task_name).mkdir(parents=True, exist_ok=True)

    info = {"description": "Plan and Preprocess nnUNet", "client_category": "Executor", "controller_type": "server"}

    meta = {
        "name": f"{task_name}_nnUNet",
        "resource_spec": {},
        "deploy_map": {f"{task_name}-server": ["server"]},
        "min_clients": 1,
        "mandatory_clients": list(clients.keys()),
    }
    for client_id in clients:
        meta["deploy_map"][f"{task_name}-client-{client_id}"] = [client_id]

    with open(Path(root_dir).joinpath(task_name).joinpath("info.conf"), "w") as f:
        f.write("{\n")
        f.write(HOCONConverter.to_hocon(ConfigFactory.from_dict(info)))
        f.write("\n}")

    with open(Path(root_dir).joinpath(task_name).joinpath("meta.conf"), "w") as f:
        f.write("{\n")
        f.write(HOCONConverter.to_hocon(ConfigFactory.from_dict(meta)))
        f.write("\n}")

    server = {
        "format_version": 2,
        "server": {"heart_beat_timeout": 600},
        "task_data_filters": [],
        "task_result_filters": [],
        "components": [
            {"id": "nnunet_processor", "path": "monai.nvflare.response_processor.nnUNetPlanProcessor", "args": {}},
            {"id": "json_generator", "path": "monai.nvflare.json_generator.nnUNetPlansJsonGenerator", "args": {}},
        ],
        "workflows": [
            {
                "id": "broadcast_and_process",
                "name": "BroadcastAndProcess",
                "args": {
                    "processor": "nnunet_processor",
                    "min_responses_required": 0,
                    "wait_time_after_min_received": 10,
                    "task_name": task_name,
                    "timeout": 6000,
                },
            }
        ],
    }
    Path(root_dir).joinpath(task_name).joinpath(f"{task_name}-server").mkdir(parents=True, exist_ok=True)
    with open(Path(root_dir).joinpath(task_name).joinpath(f"{task_name}-server", "config_fed_server.conf"), "w") as f:
        f.write("{\n")
        f.write(HOCONConverter.to_hocon(ConfigFactory.from_dict(server)))
        f.write("\n}")

    for client_id in clients:
        client = {
            "format_version": 2,
            "task_result_filters": [],
            "task_data_filters": [],
            "components": [],
            "executors": [
                {
                    "tasks": [task_name],
                    "executor": {
                        "path": "monai.nvflare.nnunet_executor.nnUNetExecutor",
                        "args": {
                            "data_dir": clients[client_id]["data_dir"],
                            "patient_id_in_file_identifier": clients[client_id]["patient_id_in_file_identifier"],
                            "modality_dict": clients[client_id]["modality_dict"],
                            "dataset_format": clients[client_id]["dataset_format"],
                            "nnunet_root_folder": clients[client_id]["nnunet_root_folder"],
                            "nnunet_config": {
                                "dataset_name_or_id": experiment["dataset_name_or_id"],
                                "experiment_name": experiment["experiment_name"],
                            },
                            "client_name": clients[client_id]["client_name"],
                            "tracking_uri": experiment["tracking_uri"],
                        },
                    },
                }
            ],
        }

        if "nnunet_plans" in experiment:
            client["executors"][0]["executor"]["args"]["nnunet_config"]["nnunet_plans"] = experiment["nnunet_plans"]

        if "nnunet_trainer" in experiment:
            client["executors"][0]["executor"]["args"]["nnunet_config"]["nnunet_trainer"] = experiment["nnunet_trainer"]

        Path(root_dir).joinpath(task_name).joinpath(f"{task_name}-client-{client_id}").mkdir(
            parents=True, exist_ok=True
        )
        with open(
            Path(root_dir).joinpath(task_name).joinpath(f"{task_name}-client-{client_id}", "config_fed_client.conf"),
            "w",
        ) as f:
            f.write("{\n")
            f.write(HOCONConverter.to_hocon(ConfigFactory.from_dict(client)))
            f.write("\n}")

    subprocess.run(
        [
            nvflare_exec,
            "job",
            "create",
            "-j",
            Path(root_dir).joinpath("jobs", task_name),
            "-w",
            Path(root_dir).joinpath(task_name),
            "-sd",
            script_dir,
            "--force",
        ]
    )


def preprocess_config(clients, experiment, root_dir, script_dir, nvflare_exec):
    """
    Generate job configuration files for the preprocessing task in NVFlare.

    Parameters
    ----------
    clients : dict
        A dictionary containing client-specific configurations. Each key is a client ID, and the value is a dictionary
        with the following keys:
        - 'data_dir': str, path to the client's data directory.
        - 'patient_id_in_file_identifier': str, identifier for patient ID in the file.
        - 'modality_dict': dict, dictionary mapping modalities.
        - 'dataset_format': str, format of the dataset.
        - 'nnunet_root_folder': str, root folder for nnUNet.
        - 'client_name': str, name of the client.
    experiment : dict
        A dictionary containing experiment-specific configurations with the following keys:
        - 'dataset_name_or_id': str, name or ID of the dataset.
        - 'experiment_name': str, name of the experiment.
        - 'tracking_uri': str, URI for tracking.
        - 'nnunet_plans' (optional): str, nnUNet plans.
        - 'nnunet_trainer' (optional): str, nnUNet trainer.
    root_dir : str
        The root directory where the configuration files will be generated.
    script_dir : str
        The directory containing the scripts to be used in the job.
    nvflare_exec : str
        The NVFlare executable to be used for creating the job.

    Returns
    -------
    None
    """
    task_name = "preprocess"
    Path(root_dir).joinpath(task_name).mkdir(parents=True, exist_ok=True)

    info = {"description": "Preprocess nnUNet", "client_category": "Executor", "controller_type": "server"}

    meta = {
        "name": f"{task_name}_nnUNet",
        "resource_spec": {},
        "deploy_map": {f"{task_name}-server": ["server"]},
        "min_clients": 1,
        "mandatory_clients": list(clients.keys()),
    }
    for client_id in clients:
        meta["deploy_map"][f"{task_name}-client-{client_id}"] = [client_id]

    with open(Path(root_dir).joinpath(task_name).joinpath("info.conf"), "w") as f:
        f.write("{\n")
        f.write(HOCONConverter.to_hocon(ConfigFactory.from_dict(info)))
        f.write("\n}")

    with open(Path(root_dir).joinpath(task_name).joinpath("meta.conf"), "w") as f:
        f.write("{\n")
        f.write(HOCONConverter.to_hocon(ConfigFactory.from_dict(meta)))
        f.write("\n}")

    server = {
        "format_version": 2,
        "server": {"heart_beat_timeout": 600},
        "task_data_filters": [],
        "task_result_filters": [],
        "components": [
            {"id": "nnunet_processor", "path": "monai.nvflare.response_processor.nnUNetPlanProcessor", "args": {}}
        ],
        "workflows": [
            {
                "id": "broadcast_and_process",
                "name": "BroadcastAndProcess",
                "args": {
                    "processor": "nnunet_processor",
                    "min_responses_required": 0,
                    "wait_time_after_min_received": 10,
                    "task_name": task_name,
                    "timeout": 6000,
                },
            }
        ],
    }
    Path(root_dir).joinpath(task_name).joinpath(f"{task_name}-server").mkdir(parents=True, exist_ok=True)
    with open(Path(root_dir).joinpath(task_name).joinpath(f"{task_name}-server", "config_fed_server.conf"), "w") as f:
        f.write("{\n")
        f.write(HOCONConverter.to_hocon(ConfigFactory.from_dict(server)))
        f.write("\n}")

    for client_id in clients:
        client = {
            "format_version": 2,
            "task_result_filters": [],
            "task_data_filters": [],
            "components": [],
            "executors": [
                {
                    "tasks": [task_name],
                    "executor": {
                        "path": "monai.nvflare.nnunet_executor.nnUNetExecutor",
                        "args": {
                            "data_dir": clients[client_id]["data_dir"],
                            "patient_id_in_file_identifier": clients[client_id]["patient_id_in_file_identifier"],
                            "modality_dict": clients[client_id]["modality_dict"],
                            "dataset_format": clients[client_id]["dataset_format"],
                            "nnunet_root_folder": clients[client_id]["nnunet_root_folder"],
                            "nnunet_config": {
                                "dataset_name_or_id": experiment["dataset_name_or_id"],
                                "experiment_name": experiment["experiment_name"],
                            },
                            "client_name": clients[client_id]["client_name"],
                            "tracking_uri": experiment["tracking_uri"],
                        },
                    },
                }
            ],
        }

        if "nnunet_plans" in experiment:
            client["executors"][0]["executor"]["args"]["nnunet_config"]["nnunet_plans"] = experiment["nnunet_plans"]

        if "nnunet_trainer" in experiment:
            client["executors"][0]["executor"]["args"]["nnunet_config"]["nnunet_trainer"] = experiment["nnunet_trainer"]

        Path(root_dir).joinpath(task_name).joinpath(f"{task_name}-client-{client_id}").mkdir(
            parents=True, exist_ok=True
        )
        with open(
            Path(root_dir).joinpath(task_name).joinpath(f"{task_name}-client-{client_id}", "config_fed_client.conf"),
            "w",
        ) as f:
            f.write("{\n")
            f.write(HOCONConverter.to_hocon(ConfigFactory.from_dict(client)))
            f.write("\n}")

    subprocess.run(
        [
            nvflare_exec,
            "job",
            "create",
            "-j",
            Path(root_dir).joinpath("jobs", task_name),
            "-w",
            Path(root_dir).joinpath(task_name),
            "-sd",
            script_dir,
            "--force",
        ]
    )


def train_config(clients, experiment, root_dir, script_dir, nvflare_exec):
    """
    Generate training configuration files for nnUNet using NVFlare.

    Parameters
    ----------
    clients : dict
        Dictionary containing client-specific configurations. Each key is a client ID, and the value is a dictionary
        with the following keys:
        - 'data_dir': str, path to the client's data directory.
        - 'patient_id_in_file_identifier': str, identifier for patient ID in file.
        - 'modality_dict': dict, dictionary mapping modalities.
        - 'dataset_format': str, format of the dataset.
        - 'nnunet_root_folder': str, path to the nnUNet root folder.
        - 'client_name': str, name of the client.
        - 'bundle_root': str, optional, path to the bundle root directory.
    experiment : dict
        Dictionary containing experiment-specific configurations with the following keys:
        - 'dataset_name_or_id': str, name or ID of the dataset.
        - 'experiment_name': str, name of the experiment.
        - 'tracking_uri': str, URI for tracking.
        - 'nnunet_plans': str, optional, nnUNet plans.
        - 'nnunet_trainer': str, optional, nnUNet trainer.
    root_dir : str
        Root directory where the configuration files will be generated.
    script_dir : str
        Directory containing the scripts to be used.
    nvflare_exec : str
        Path to the NVFlare executable.

    Returns
    -------
    None
    """
    task_name = "train"
    Path(root_dir).joinpath(task_name).mkdir(parents=True, exist_ok=True)

    info = {"description": "Train nnUNet", "client_category": "Executor", "controller_type": "server"}

    meta = {
        "name": f"{task_name}_nnUNet",
        "resource_spec": {},
        "deploy_map": {f"{task_name}-server": ["server"]},
        "min_clients": 1,
        "mandatory_clients": list(clients.keys()),
    }
    for client_id in clients:
        meta["deploy_map"][f"{task_name}-client-{client_id}"] = [client_id]

    with open(Path(root_dir).joinpath(task_name).joinpath("info.conf"), "w") as f:
        f.write("{\n")
        f.write(HOCONConverter.to_hocon(ConfigFactory.from_dict(info)))
        f.write("\n}")

    with open(Path(root_dir).joinpath(task_name).joinpath("meta.conf"), "w") as f:
        f.write("{\n")
        f.write(HOCONConverter.to_hocon(ConfigFactory.from_dict(meta)))
        f.write("\n}")

    server = {
        "format_version": 2,
        "server": {"heart_beat_timeout": 600},
        "task_data_filters": [],
        "task_result_filters": [],
        "components": [
            {"id": "nnunet_processor", "path": "monai.nvflare.response_processor.nnUNetTrainProcessor", "args": {}},
            {"id": "json_generator", "path": "monai.nvflare.json_generator.nnUNetValSummaryJsonGenerator", "args": {}},
        ],
        "workflows": [
            {
                "id": "broadcast_and_process",
                "name": "BroadcastAndProcess",
                "args": {
                    "processor": "nnunet_processor",
                    "min_responses_required": 0,
                    "wait_time_after_min_received": 10,
                    "task_name": task_name,
                    "timeout": 600000,
                },
            }
        ],
    }
    Path(root_dir).joinpath(task_name).joinpath(f"{task_name}-server").mkdir(parents=True, exist_ok=True)
    with open(Path(root_dir).joinpath(task_name).joinpath(f"{task_name}-server", "config_fed_server.conf"), "w") as f:
        f.write("{\n")
        f.write(HOCONConverter.to_hocon(ConfigFactory.from_dict(server)))
        f.write("\n}")

    for client_id in clients:
        client = {
            "format_version": 2,
            "task_result_filters": [],
            "task_data_filters": [],
            "components": [],
            "executors": [
                {
                    "tasks": [task_name],
                    "executor": {
                        "path": "monai.nvflare.nnunet_executor.nnUNetExecutor",
                        "args": {
                            "data_dir": clients[client_id]["data_dir"],
                            "patient_id_in_file_identifier": clients[client_id]["patient_id_in_file_identifier"],
                            "modality_dict": clients[client_id]["modality_dict"],
                            "dataset_format": clients[client_id]["dataset_format"],
                            "nnunet_root_folder": clients[client_id]["nnunet_root_folder"],
                            "nnunet_config": {
                                "dataset_name_or_id": experiment["dataset_name_or_id"],
                                "experiment_name": experiment["experiment_name"],
                            },
                            "client_name": clients[client_id]["client_name"],
                            "tracking_uri": experiment["tracking_uri"]
                        },
                    },
                }
            ],
        }
        if "continue_training" in experiment:
            client["executors"][0]["executor"]["args"]["continue_training"] = experiment["continue_training"]
        if "nnunet_plans" in experiment:
            client["executors"][0]["executor"]["args"]["nnunet_config"]["nnunet_plans"] = experiment["nnunet_plans"]

        if "nnunet_trainer" in experiment:
            client["executors"][0]["executor"]["args"]["nnunet_config"]["nnunet_trainer"] = experiment["nnunet_trainer"]

        if "bundle_root" in clients[client_id]:
            client["executors"][0]["executor"]["args"]["bundle_root"] = clients[client_id]["bundle_root"]

        Path(root_dir).joinpath(task_name).joinpath(f"{task_name}-client-{client_id}").mkdir(
            parents=True, exist_ok=True
        )
        with open(
            Path(root_dir).joinpath(task_name).joinpath(f"{task_name}-client-{client_id}", "config_fed_client.conf"),
            "w",
        ) as f:
            f.write("{\n")
            f.write(HOCONConverter.to_hocon(ConfigFactory.from_dict(client)))
            f.write("\n}")

    subprocess.run(
        [
            nvflare_exec,
            "job",
            "create",
            "-j",
            Path(root_dir).joinpath("jobs", task_name),
            "-w",
            Path(root_dir).joinpath(task_name),
            "-sd",
            script_dir,
            "--force",
        ]
    )


def prepare_bundle_config(clients, experiment, root_dir, script_dir, nvflare_exec):
    """
    Prepare the configuration files for the nnUNet bundle and generate the job configurations for NVFlare.

    Parameters
    ----------
    clients : dict
        A dictionary containing client information. Keys are client IDs and values are dictionaries with client details.
    experiment : dict
        A dictionary containing experiment details such as 'experiment_name', 'tracking_uri', and optional
        configurations like 'bundle_extra_config', 'nnunet_plans', and 'nnunet_trainer'.
    root_dir : str
        The root directory where the configuration files and job directories will be created.
    script_dir : str
        The directory containing the necessary scripts for NVFlare.
    nvflare_exec : str
        The path to the NVFlare executable.

    Returns
    -------
    None
    """
    task_name = "prepare_bundle"
    Path(root_dir).joinpath(task_name).mkdir(parents=True, exist_ok=True)

    info = {"description": "Prepare nnUNet Bundle", "client_category": "Executor", "controller_type": "server"}

    meta = {
        "name": f"{task_name}_nnUNet",
        "resource_spec": {},
        "deploy_map": {f"{task_name}-server": ["server"]},
        "min_clients": 1,
        "mandatory_clients": list(clients.keys()),
    }
    for client_id in clients:
        meta["deploy_map"][f"{task_name}-client-{client_id}"] = [client_id]

    with open(Path(root_dir).joinpath(task_name).joinpath("info.conf"), "w") as f:
        f.write("{\n")
        f.write(HOCONConverter.to_hocon(ConfigFactory.from_dict(info)))
        f.write("\n}")

    with open(Path(root_dir).joinpath(task_name).joinpath("meta.conf"), "w") as f:
        f.write("{\n")
        f.write(HOCONConverter.to_hocon(ConfigFactory.from_dict(meta)))
        f.write("\n}")

    server = {
        "format_version": 2,
        "server": {"heart_beat_timeout": 600},
        "task_data_filters": [],
        "task_result_filters": [],
        "components": [
            {
                "id": "nnunet_processor",
                "path": "monai.nvflare.response_processor.nnUNetBundlePrepareProcessor",
                "args": {},
            }
        ],
        "workflows": [
            {
                "id": "broadcast_and_process",
                "name": "BroadcastAndProcess",
                "args": {
                    "processor": "nnunet_processor",
                    "min_responses_required": 0,
                    "wait_time_after_min_received": 10,
                    "task_name": task_name,
                    "timeout": 600000,
                },
            }
        ],
    }
    Path(root_dir).joinpath(task_name).joinpath(f"{task_name}-server").mkdir(parents=True, exist_ok=True)
    with open(Path(root_dir).joinpath(task_name).joinpath(f"{task_name}-server", "config_fed_server.conf"), "w") as f:
        f.write("{\n")
        f.write(HOCONConverter.to_hocon(ConfigFactory.from_dict(server)))
        f.write("\n}")

    for client_id in clients:
        client = {
            "format_version": 2,
            "task_result_filters": [],
            "task_data_filters": [],
            "components": [],
            "executors": [
                {
                    "tasks": [task_name],
                    "executor": {
                        "path": "monai.nvflare.nnunet_executor.nnUNetExecutor",
                        "args": {
                            "nnunet_config": {"experiment_name": experiment["experiment_name"]},
                            "client_name": clients[client_id]["client_name"],
                            "tracking_uri": experiment["tracking_uri"],
                        },
                    },
                }
            ],
        }

        if "bundle_extra_config" in experiment:
            client["executors"][0]["executor"]["args"]["train_extra_configs"] = experiment["bundle_extra_config"]
        if "nnunet_plans" in experiment:
            client["executors"][0]["executor"]["args"]["nnunet_config"]["nnunet_plans"] = experiment["nnunet_plans"]

        if "nnunet_trainer" in experiment:
            client["executors"][0]["executor"]["args"]["nnunet_config"]["nnunet_trainer"] = experiment["nnunet_trainer"]

        if "bundle_root" in clients[client_id]:
            client["executors"][0]["executor"]["args"]["bundle_root"] = clients[client_id]["bundle_root"]

        Path(root_dir).joinpath(task_name).joinpath(f"{task_name}-client-{client_id}").mkdir(
            parents=True, exist_ok=True
        )
        with open(
            Path(root_dir).joinpath(task_name).joinpath(f"{task_name}-client-{client_id}", "config_fed_client.conf"),
            "w",
        ) as f:
            f.write("{\n")
            f.write(HOCONConverter.to_hocon(ConfigFactory.from_dict(client)))
            f.write("\n}")

    subprocess.run(
        [
            nvflare_exec,
            "job",
            "create",
            "-j",
            Path(root_dir).joinpath("jobs", task_name),
            "-w",
            Path(root_dir).joinpath(task_name),
            "-sd",
            script_dir,
            "--force",
        ]
    )


def train_fl_config(clients, experiment, root_dir, script_dir, nvflare_exec):
    """
    Generate federated learning job configurations for NVFlare.

    Parameters
    ----------
    clients : dict
        Dictionary containing client names and their configurations.
    experiment : dict
        Dictionary containing experiment parameters such as number of rounds and local epochs.
    root_dir : str
        Root directory where the job configurations will be saved.
    script_dir : str
        Directory containing the necessary scripts for NVFlare.
    nvflare_exec : str
        Path to the NVFlare executable.

    Returns
    -------
    None
    """
    task_name = "train_fl_nnunet_bundle"
    Path(root_dir).joinpath(task_name).mkdir(parents=True, exist_ok=True)

    info = {
        "description": "Federated Learning with nnUNet-MONAI Bundle",
        "client_category": "Executor",
        "controller_type": "server",
    }

    meta = {
        "name": f"{task_name}",
        "resource_spec": {},
        "deploy_map": {f"{task_name}-server": ["server"]},
        "min_clients": len(list(clients.keys())),
        "mandatory_clients": list(clients.keys()),
    }

    for client_name, client_config in clients.items():
        meta["deploy_map"][f"{task_name}-{client_name}"] = [client_name]

    with open(Path(root_dir).joinpath(task_name).joinpath("info.conf"), "w") as f:
        f.write("{\n")
        f.write(HOCONConverter.to_hocon(ConfigFactory.from_dict(info)))
        f.write("\n}")

    with open(Path(root_dir).joinpath(task_name).joinpath("meta.conf"), "w") as f:
        f.write("{\n")
        f.write(HOCONConverter.to_hocon(ConfigFactory.from_dict(meta)))
        f.write("\n}")

    server = {
        "format_version": 2,
        "min_clients": len(list(clients.keys())),
        "num_rounds": experiment["num_rounds"],
        "task_data_filters": [],
        "task_result_filters": [],
        "components": [
            {
                "id": "persistor",
                "path": "monai_nvflare.monai_bundle_persistor.MonaiBundlePersistor",
                "args": {
                    "bundle_root": experiment["server_bundle_root"],
                    "config_train_filename": "configs/train.yaml",
                    "network_def_key": "network_def_fl",
                },
            },
            {"id": "shareable_generator", "name": "FullModelShareableGenerator", "args": {}},
            {
                "id": "aggregator",
                "name": "InTimeAccumulateWeightedAggregator",
                "args": {"expected_data_kind": "WEIGHT_DIFF"},
            },
            {"id": "model_selector", "name": "IntimeModelSelector", "args": {}},
            {"id": "model_locator", "name": "PTFileModelLocator", "args": {"pt_persistor_id": "persistor"}},
            {"id": "json_generator", "name": "ValidationJsonGenerator", "args": {}},
        ],
        "workflows": [
            {
                "id": "scatter_gather_ctl",
                "name": "ScatterAndGather",
                "args": {
                    "min_clients": "{min_clients}",
                    "num_rounds": "{num_rounds}",
                    "start_round": experiment["start_round"],
                    "wait_time_after_min_received": 10,
                    "aggregator_id": "aggregator",
                    "persistor_id": "persistor",
                    "shareable_generator_id": "shareable_generator",
                    "train_task_name": "train",
                    "train_timeout": 0,
                },
            },
            {
                "id": "cross_site_model_eval",
                "name": "CrossSiteModelEval",
                "args": {
                    "model_locator_id": "model_locator",
                    "submit_model_timeout": 600,
                    "validation_timeout": 6000,
                    "cleanup_models": True,
                },
            },
        ],
    }
    Path(root_dir).joinpath(task_name).joinpath(f"{task_name}-server").mkdir(parents=True, exist_ok=True)
    with open(Path(root_dir).joinpath(task_name).joinpath(f"{task_name}-server", "config_fed_server.conf"), "w") as f:
        f.write("{\n")
        f.write(HOCONConverter.to_hocon(ConfigFactory.from_dict(server)))
        f.write("\n}")

    for client_name, client_config in clients.items():
        client = {
            "format_version": 2,
            "task_result_filters": [],
            "task_data_filters": [],
            "executors": [
                {
                    "tasks": ["train", "submit_model", "validate"],
                    "executor": {
                        "id": "executor",
                        # "path": "monai_algo.ClientnnUNetAlgoExecutor",
                        "path": "monai_nvflare.client_algo_executor.ClientAlgoExecutor",
                        "args": {"client_algo_id": "client_algo", "key_metric": "Val_Dice"},
                    },
                }
            ],
            "components": [
                {
                    "id": "client_algo",
                    # "path": "monai_algo.MonaiAlgonnUNet",
                    "path": "monai.fl.client.monai_algo.MonaiAlgo",
                    "args": {
                        "bundle_root": client_config["bundle_root"],
                        "config_train_filename": "configs/train.yaml",
                        "save_dict_key": "network_weights",
                        "local_epochs": experiment["local_epochs"],
                        "train_kwargs": {"nnunet_root_folder": client_config["nnunet_root_folder"]},
                    },
                }
            ],
        }

        Path(root_dir).joinpath(task_name).joinpath(f"{task_name}-{client_name}").mkdir(parents=True, exist_ok=True)
        with open(
            Path(root_dir).joinpath(task_name).joinpath(f"{task_name}-{client_name}", "config_fed_client.conf"), "w"
        ) as f:
            f.write("{\n")
            f.write(HOCONConverter.to_hocon(ConfigFactory.from_dict(client)))
            f.write("\n}")

    subprocess.run(
        [
            nvflare_exec,
            "job",
            "create",
            "-j",
            Path(root_dir).joinpath("jobs", task_name),
            "-w",
            Path(root_dir).joinpath(task_name),
            "-sd",
            script_dir,
            "--force",
        ]
    )


def generate_configs(client_files, experiment_file, script_dir, job_dir, nvflare_exec="nvflare"):
    """
    Generate configuration files for NVFlare job.

    Parameters
    ----------
    client_files : list of str
        List of file paths to client configuration files.
    experiment_file : str
        File path to the experiment configuration file.
    script_dir : str
        Directory path where the scripts are located.
    job_dir : str
        Directory path where the job configurations will be saved.
    nvflare_exec : str, optional
        NVFlare executable command, by default "nvflare".

    Returns
    -------
    None
    """
    clients = {}
    for client_id in client_files:
        with open(client_id) as f:
            client_name = Path(client_id).name
            clients[client_name.split(".")[0]] = yaml.safe_load(f)

    with open(experiment_file) as f:
        experiment = yaml.safe_load(f)

    check_client_packages_config(clients, experiment, job_dir, script_dir, nvflare_exec)
    prepare_config(clients, experiment, job_dir, script_dir, nvflare_exec)
    plan_and_preprocess_config(clients, experiment, job_dir, script_dir, nvflare_exec)
    preprocess_config(clients, experiment, job_dir, script_dir, nvflare_exec)
    train_config(clients, experiment, job_dir, script_dir, nvflare_exec)
    prepare_bundle_config(clients, experiment, job_dir, script_dir, nvflare_exec)
    train_fl_config(clients, experiment, job_dir, script_dir, nvflare_exec)

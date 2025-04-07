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

import json
import logging
import multiprocessing
import os
import pathlib
import random
import re
import shutil
import subprocess
from importlib.metadata import version
from pathlib import Path

import torch
import mlflow
import numpy as np
import pandas as pd
import psutil
import yaml

import monai
from monai.apps.nnunet import nnUNetV2Runner
from monai.apps.nnunet.nnunet_bundle import convert_monai_bundle_to_nnunet
from monai.bundle import ConfigParser


def train(
    nnunet_root_dir,
    experiment_name,
    client_name,
    tracking_uri,
    dataset_name_or_id,
    trainer_class_name="nnUNetTrainer",
    nnunet_plans_name="nnUNetPlans",
    run_with_bundle=False,
    fold=0,
    bundle_root=None,
    mlflow_token=None,
    continue_training=False,
    resume_epoch="latest",
):
    """

    Train a nnUNet model and log metrics to MLflow.

    Parameters
    ----------
    nnunet_root_dir : str
        Root directory for nnUNet.
    experiment_name : str
        Name of the MLflow experiment.
    client_name : str
        Name of the client.
    tracking_uri : str
        URI for MLflow tracking server.
    dataset_name_or_id : str
        Name or ID of the dataset.
    trainer_class_name : str, optional
        Name of the nnUNet trainer class, by default "nnUNetTrainer".
    nnunet_plans_name : str, optional
        Name of the nnUNet plans, by default "nnUNetPlans".
    run_with_bundle : bool, optional
        Whether to run with MONAI bundle, by default False.
    fold : int, optional
        Fold number for cross-validation, by default 0.
    bundle_root : str, optional
        Root directory for MONAI bundle, by default None.
    mlflow_token : str, optional
        Token for MLflow authentication, by default None.
    continue_training : bool, optional
        Whether to continue training from a checkpoint, by default False.
    resume_epoch : int, optional
        Epoch to resume training from, by default "latest".

    Returns
    -------
    dict
        Dictionary containing validation summary metrics.
    """
    data_src_cfg = os.path.join(nnunet_root_dir, f"Task{dataset_name_or_id}_data_src_cfg.yaml")
    runner = nnUNetV2Runner(input_config=data_src_cfg, trainer_class_name=trainer_class_name, work_dir=nnunet_root_dir)

    if not run_with_bundle:
        if continue_training:
            runner.train_single_model(config="3d_fullres", fold=fold, c=True)
        else:
            runner.train_single_model(config="3d_fullres", fold=fold)
    else:
        os.environ["BUNDLE_ROOT"] = bundle_root
        os.environ["PYTHONPATH"] = os.environ["PYTHONPATH"] + ":" + bundle_root
        config_files = os.path.join(bundle_root, "configs", "train.yaml")
        if continue_training:
            config_files = [os.path.join(bundle_root, "configs", "train.yaml"), os.path.join(bundle_root, "configs", "train_resume.yaml")]
        monai.bundle.run(
            config_file=config_files,
            bundle_root=bundle_root,
            nnunet_trainer_class_name=trainer_class_name,
            mlflow_experiment_name=experiment_name,
            mlflow_run_name="run_" + client_name,
            tracking_uri=tracking_uri,
            fold_id=fold,
            nnunet_root_folder=nnunet_root_dir,
            reload_checkpoint_epoch=resume_epoch
        )
        nnunet_config = {"dataset_name_or_id": dataset_name_or_id, "nnunet_trainer": trainer_class_name}
        convert_monai_bundle_to_nnunet(nnunet_config, bundle_root)
        runner.train_single_model(config="3d_fullres", fold=fold, val="")

    if mlflow_token is not None:
        os.environ["MLFLOW_TRACKING_TOKEN"] = mlflow_token
    if tracking_uri is not None:
        mlflow.set_tracking_uri(tracking_uri)

    try:
        mlflow.create_experiment(experiment_name)
    except Exception as e:
        print(e)
        mlflow.set_experiment(experiment_id=(mlflow.get_experiment_by_name(experiment_name).experiment_id))

    filter = f"""
    tags."client" = "{client_name}"
    """

    runs = mlflow.search_runs(experiment_names=[experiment_name], filter_string=filter, order_by=["start_time DESC"])

    validation_summary = os.path.join(
        runner.nnunet_results,
        runner.dataset_name,
        f"{trainer_class_name}__{nnunet_plans_name}__3d_fullres",
        f"fold_{fold}",
        "validation",
        "summary.json",
    )

    dataset_file = os.path.join(
        runner.nnunet_results,
        runner.dataset_name,
        f"{trainer_class_name}__{nnunet_plans_name}__3d_fullres",
        "dataset.json",
    )

    with open(dataset_file, "r") as f:
        dataset_dict = json.load(f)
        labels = dataset_dict["labels"]
        labels = {str(v): k for k, v in labels.items()}

    with open(validation_summary, "r") as f:
        validation_summary_dict = json.load(f)

    if len(runs) == 0:
        with mlflow.start_run(run_name=f"run_{client_name}", tags={"client": client_name}):
            for label in validation_summary_dict["mean"]:
                for metric in validation_summary_dict["mean"][label]:
                    label_name = labels[label]
                    mlflow.log_metric(f"{label_name}_{metric}", float(validation_summary_dict["mean"][label][metric]))

    else:
        with mlflow.start_run(run_id=runs.iloc[0].run_id, tags={"client": client_name}):
            for label in validation_summary_dict["mean"]:
                for metric in validation_summary_dict["mean"][label]:
                    label_name = labels[label]
                    mlflow.log_metric(f"{label_name}_{metric}", float(validation_summary_dict["mean"][label][metric]))

    return validation_summary_dict


def preprocess(nnunet_root_dir, dataset_name_or_id, nnunet_plans_file_path=None, trainer_class_name="nnUNetTrainer"):
    """
    Preprocess the dataset for nnUNet training.

    Parameters
    ----------
    nnunet_root_dir : str
        The root directory of the nnUNet project.
    dataset_name_or_id : str or int
        The name or ID of the dataset to preprocess.
    nnunet_plans_file_path : Path, optional
        The file path to the nnUNet plans file. If None, default plans will be used. Default is None.
    trainer_class_name : str, optional
        The name of the trainer class to use. Default is "nnUNetTrainer".

    Returns
    -------
    dict
        The nnUNet plans dictionary.
    """

    data_src_cfg = os.path.join(nnunet_root_dir, f"Task{dataset_name_or_id}_data_src_cfg.yaml")
    runner = nnUNetV2Runner(input_config=data_src_cfg, trainer_class_name=trainer_class_name, work_dir=nnunet_root_dir)

    nnunet_plans_name = nnunet_plans_file_path.name.split(".")[0]
    from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name

    dataset_name = maybe_convert_to_dataset_name(int(dataset_name_or_id))

    Path(nnunet_root_dir).joinpath("nnUNet_preprocessed", dataset_name).mkdir(parents=True, exist_ok=True)

    shutil.copy(
        Path(nnunet_root_dir).joinpath("nnUNet_raw_data_base", dataset_name, "dataset.json"),
        Path(nnunet_root_dir).joinpath("nnUNet_preprocessed", dataset_name, "dataset.json"),
    )
    if nnunet_plans_file_path is not None:
        with open(nnunet_plans_file_path, "r") as f:
            nnunet_plans = json.load(f)
            nnunet_plans["original_dataset_name"] = nnunet_plans["dataset_name"]
            nnunet_plans["dataset_name"] = dataset_name
            json.dump(
                nnunet_plans,
                open(
                    Path(nnunet_root_dir).joinpath("nnUNet_preprocessed", dataset_name, f"{nnunet_plans_name}.json"),
                    "w",
                ),
                indent=4,
            )

    runner.extract_fingerprints(npfp=2, verify_dataset_integrity=True)
    runner.preprocess(c=["3d_fullres"], n_proc=[2], overwrite_plans_name=nnunet_plans_name)

    return nnunet_plans


def plan_and_preprocess(
    nnunet_root_dir,
    dataset_name_or_id,
    client_name,
    experiment_name,
    tracking_uri,
    mlflow_token=None,
    nnunet_plans_name="nnUNetPlans",
    trainer_class_name="nnUNetTrainer",
):
    """
    Plan and preprocess the dataset using nnUNetV2Runner and log the plans to MLflow.

    Parameters
    ----------
    nnunet_root_dir : str
        The root directory of nnUNet.
    dataset_name_or_id : str or int
        The name or ID of the dataset to be processed.
    client_name : str
        The name of the client.
    experiment_name : str
        The name of the MLflow experiment.
    tracking_uri : str
        The URI of the MLflow tracking server.
    mlflow_token : str, optional
        The token for MLflow authentication (default is None).
    nnunet_plans_name : str, optional
        The name of the nnUNet plans (default is "nnUNetPlans").
    trainer_class_name : str, optional
        The name of the nnUNet trainer class (default is "nnUNetTrainer").

    Returns
    -------
    dict
        The nnUNet plans as a dictionary.
    """

    data_src_cfg = os.path.join(nnunet_root_dir, f"Task{dataset_name_or_id}_data_src_cfg.yaml")

    runner = nnUNetV2Runner(input_config=data_src_cfg, trainer_class_name=trainer_class_name, work_dir=nnunet_root_dir)

    runner.plan_and_process(
        npfp=2, verify_dataset_integrity=True, c=["3d_fullres"], n_proc=[2], overwrite_plans_name=nnunet_plans_name
    )

    preprocessed_folder = runner.nnunet_preprocessed

    from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name

    dataset_name = maybe_convert_to_dataset_name(int(dataset_name_or_id))

    with open(Path(preprocessed_folder).joinpath(f"{dataset_name}", nnunet_plans_name + ".json"), "r") as f:
        nnunet_plans = json.load(f)

    if mlflow_token is not None:
        os.environ["MLFLOW_TRACKING_TOKEN"] = mlflow_token
    if tracking_uri is not None:
        mlflow.set_tracking_uri(tracking_uri)

    try:
        mlflow.create_experiment(experiment_name)
    except Exception as e:
        print(e)
        mlflow.set_experiment(experiment_id=(mlflow.get_experiment_by_name(experiment_name).experiment_id))

    filter = f"""
    tags."client" = "{client_name}"
    """

    runs = mlflow.search_runs(experiment_names=[experiment_name], filter_string=filter, order_by=["start_time DESC"])

    if len(runs) == 0:
        with mlflow.start_run(run_name=f"run_{client_name}", tags={"client": client_name}):
            mlflow.log_dict(nnunet_plans, nnunet_plans_name + ".json")

    else:
        with mlflow.start_run(run_id=runs.iloc[0].run_id, tags={"client": client_name}):
            mlflow.log_dict(nnunet_plans, nnunet_plans_name + ".json")

    return nnunet_plans


def prepare_data_folder(
    data_dir,
    nnunet_root_dir,
    dataset_name_or_id,
    modality_dict,
    experiment_name,
    client_name,
    dataset_format,
    modality_list = None,
    tracking_uri=None,
    mlflow_token=None,
    subfolder_suffix=None,
    patient_id_in_file_identifier=True,
    trainer_class_name="nnUNetTrainer",
):
    """
    Prepare the data folder for nnUNet training and log the data to MLflow.

    Parameters
    ----------
    data_dir : str
        Directory containing the dataset.
    nnunet_root_dir : str
        Root directory for nnUNet.
    dataset_name_or_id : str
        Name or ID of the dataset.
    modality_dict : dict
        Dictionary mapping modality IDs to file suffixes.
    experiment_name : str
        Name of the MLflow experiment.
    client_name : str
        Name of the client.
    dataset_format : str
        Format of the dataset. Supported formats are "subfolders", "decathlon", and "nnunet".
    tracking_uri : str, optional
        URI for MLflow tracking server.
    modality_list : list, optional
        List of modalities. Default is None.
    mlflow_token : str, optional
        Token for MLflow authentication.
    subfolder_suffix : str, optional
        Suffix for subfolder names.
    patient_id_in_file_identifier : bool, optional
        Whether patient ID is included in file identifier. Default is True.
    trainer_class_name : str, optional
        Name of the nnUNet trainer class. Default is "nnUNetTrainer".

    Returns
    -------
    dict
        Dictionary containing the training and testing data lists.
    """
    if dataset_format == "subfolders":
        if subfolder_suffix is not None:
            data_list = {
                "training": [
                    {
                        modality_id: (
                            str(
                                pathlib.Path(f.name).joinpath(
                                    f.name[: -len(subfolder_suffix)] + modality_dict[modality_id]
                                )
                            )
                            if patient_id_in_file_identifier
                            else str(pathlib.Path(f.name).joinpath(modality_dict[modality_id]))
                        )
                        for modality_id in modality_dict
                    }
                    for f in os.scandir(data_dir)
                    if f.is_dir()
                ],
                "testing": [],
            }
        else:
            data_list = {
                "training": [
                    {
                        modality_id: (
                            str(pathlib.Path(f.name).joinpath(f.name + modality_dict[modality_id]))
                            if patient_id_in_file_identifier
                            else str(pathlib.Path(f.name).joinpath(modality_dict[modality_id]))
                        )
                        for modality_id in modality_dict
                    }
                    for f in os.scandir(data_dir)
                    if f.is_dir()
                ],
                "testing": [],
            }
    elif dataset_format == "decathlon" or dataset_format == "nnunet":
        cases = []

        for f in os.scandir(Path(data_dir).joinpath("imagesTr")):
            if f.is_file():
                for modality_suffix in list(modality_dict.values()):
                    if f.name.endswith(modality_suffix) and modality_suffix != ".nii.gz":
                        cases.append(f.name[: -len(modality_suffix)])
                if len(np.unique(list(modality_dict.values()))) == 1 and ".nii.gz" in list(modality_dict.values()):
                    cases.append(f.name[: -len(".nii.gz")])
        cases = np.unique(cases)
        data_list = {
            "training": [
                {
                    modality_id: str(Path("imagesTr").joinpath(case + modality_dict[modality_id]))
                    for modality_id in modality_dict
                    if modality_id != "label"
                }
                for case in cases
            ],
            "testing": [],
        }
        for idx, case in enumerate(data_list["training"]):
            modality_id = list(modality_dict.keys())[0]
            case_id = Path(case[modality_id]).name[: -len(modality_dict[modality_id])]
            data_list["training"][idx]["label"] = str(Path("labelsTr").joinpath(case_id + modality_dict["label"]))
    else:
        raise ValueError("Dataset format not supported")

    for idx, train_case in enumerate(data_list["training"]):
        for modality_id in modality_dict:
            data_list["training"][idx][modality_id + "_is_file"] = (
                Path(data_dir).joinpath(data_list["training"][idx][modality_id]).is_file()
            )
            if "image" not in data_list["training"][idx] and modality_id != "label":
                data_list["training"][idx]["image"] = data_list["training"][idx][modality_id]
        data_list["training"][idx]["fold"] = 0

    random.seed(42)
    random.shuffle(data_list["training"])

    data_list["testing"] = [data_list["training"][0]]

    num_folds = 5
    fold_size = len(data_list["training"]) // num_folds
    for i in range(num_folds):
        for j in range(fold_size):
            data_list["training"][i * fold_size + j]["fold"] = i

    datalist_file = Path(data_dir).joinpath(f"{experiment_name}_folds.json")
    with open(datalist_file, "w", encoding="utf-8") as f:
        json.dump(data_list, f, ensure_ascii=False, indent=4)

    os.makedirs(nnunet_root_dir, exist_ok=True)

    if modality_list is None:
        modality_list = [k for k in modality_dict.keys() if k != "label"]
    
    data_src_cfg = os.path.join(nnunet_root_dir, f"Task{dataset_name_or_id}_data_src_cfg.yaml")
    data_src = {
        "modality": modality_list,
        "dataset_name_or_id": dataset_name_or_id,
        "datalist": str(datalist_file),
        "dataroot": str(data_dir),
    }

    ConfigParser.export_config_file(data_src, data_src_cfg)

    if dataset_format != "nnunet":
        runner = nnUNetV2Runner(
            input_config=data_src_cfg, trainer_class_name=trainer_class_name, work_dir=nnunet_root_dir
        )
        runner.convert_dataset()
    else:
        ...

    if mlflow_token is not None:
        os.environ["MLFLOW_TRACKING_TOKEN"] = mlflow_token
    if tracking_uri is not None:
        mlflow.set_tracking_uri(tracking_uri)

    try:
        mlflow.create_experiment(experiment_name)
        mlflow.set_experiment(experiment_id=(mlflow.get_experiment_by_name(experiment_name).experiment_id))
    except Exception as e:
        print(e)
        mlflow.set_experiment(experiment_id=(mlflow.get_experiment_by_name(experiment_name).experiment_id))

    filter = f"""
    tags."client" = "{client_name}"
    """

    runs = mlflow.search_runs(experiment_names=[experiment_name], filter_string=filter, order_by=["start_time DESC"])

    try:
        if len(runs) == 0:
            with mlflow.start_run(run_name=f"run_{client_name}", tags={"client": client_name}):
                mlflow.log_table(pd.DataFrame.from_records(data_list["training"]), f"{client_name}_train.json")
        else:
            with mlflow.start_run(run_id=runs.iloc[0].run_id, tags={"client": client_name}):
                mlflow.log_table(pd.DataFrame.from_records(data_list["training"]), f"{client_name}_train.json")
    except (BrokenPipeError, ConnectionError) as e:
        logging.error(f"Failed to log data to MLflow: {e}")

    return data_list


def check_packages(packages):
    """
    Check if the specified packages are installed and return a report.

    Parameters
    ----------
    packages : list
        A list of package names (str) or dictionaries with keys "import_name" and "package_name".

    Returns
    -------
    dict
        A dictionary where the keys are package names and the values are strings indicating whether
        the package is installed and its version if applicable.

    Examples
    --------
    >>> check_packages(["numpy", "nonexistent_package"])
    {'numpy': 'numpy 1.21.0 is installed.', 'nonexistent_package': 'nonexistent_package is not installed.'}
    >>> check_packages([{"import_name": "torch", "package_name": "torch"}])
    {'torch': 'torch 1.9.0 is installed.'}
    """
    report = {}
    for package in packages:
        try:
            if isinstance(package, dict):
                __import__(package["import_name"])
                package_version = version(package["package_name"])
                name = package["package_name"]
                print(f"{name} {package_version} is installed.")
                report[name] = f"{name} {package_version} is installed."
            else:

                __import__(package)
                package_version = version(package)
                print(f"{package} {package_version} is installed.")
                report[package] = f"{package} {package_version} is installed."

        except ImportError:
            print(f"{package} is not installed.")
            report[package] = f"{package} is not installed."

    return report


def check_host_config():
    """
    Collects and returns the host configuration details including GPU, CPU, and memory information.

    Returns
    -------
    dict
        A dictionary containing the following keys and their corresponding values:
        - Config values from `monai.config.deviceconfig.get_config_values()`
        - Optional config values from `monai.config.deviceconfig.get_optional_config_values()`
        - GPU information including number of GPUs, CUDA version, cuDNN version, and GPU names and memory
        - CPU core count
        - Total memory in GB
        - Memory usage percentage
    """
    params_dict = {}
    config_values = monai.config.deviceconfig.get_config_values()
    for k in config_values:
        params_dict[re.sub("[()]", " ", str(k))] = config_values[k]
    optional_config_values = monai.config.deviceconfig.get_optional_config_values()

    for k in optional_config_values:
        params_dict[re.sub("[()]", " ", str(k))] = optional_config_values[k]

    gpu_info = monai.config.deviceconfig.get_gpu_info()
    allowed_keys = ["Num GPUs", "Has Cuda", "CUDA Version", "cuDNN enabled", "cuDNN Version"]
    for i in range(gpu_info["Num GPUs"]):
        allowed_keys.append(f"GPU {i} Name")
        allowed_keys.append(f"GPU {i} Total memory  GB ")

    for k in gpu_info:
        if re.sub("[()]", " ", str(k)) in allowed_keys:
            params_dict[re.sub("[()]", " ", str(k))] = str(gpu_info[k])

    with open("nvidia-smi.log", "w") as f_e:
        subprocess.run("nvidia-smi", stderr=f_e, stdout=f_e)

    params_dict["CPU_Cores"] = multiprocessing.cpu_count()

    vm = psutil.virtual_memory()

    params_dict["Total Memory"] = vm.total / (1024 * 1024 * 1024)
    params_dict["Memory Used %"] = vm.percent

    return params_dict


def prepare_bundle(bundle_config, train_extra_configs=None):
    """
    Prepare the bundle configuration for training and evaluation.

    Parameters
    ----------
    bundle_config : dict
        Dictionary containing the bundle configuration. Expected keys are:
        - "bundle_root": str, root directory of the bundle.
        - "tracking_uri": str, URI for tracking.
        - "mlflow_experiment_name": str, name of the MLflow experiment.
        - "mlflow_run_name": str, name of the MLflow run.
        - "nnunet_plans_identifier": str, optional, identifier for nnUNet plans.
        - "nnunet_trainer_class_name": str, optional, class name for nnUNet trainer.
    train_extra_configs : dict, optional
        Additional configurations for training. If provided, expected keys are:
        - "resume_epoch": int, epoch to resume training from.
        - Any other key-value pairs to be added to the training configuration.

    Returns
    -------
    None
    """

    with open(Path(bundle_config["bundle_root"]).joinpath("configs", "train.yaml")) as f:
        train_config = yaml.safe_load(f)
        train_config["bundle_root"] = bundle_config["bundle_root"]
        train_config["tracking_uri"] = bundle_config["tracking_uri"]
        train_config["mlflow_experiment_name"] = bundle_config["mlflow_experiment_name"]
        train_config["mlflow_run_name"] = bundle_config["mlflow_run_name"]

        train_config["data_src_cfg"] = "$@nnunet_root_folder+'/data_src_cfg.yaml'"
        train_config["nnunet_root_folder"] = "."
        train_config["runner"] = {
            "_target_": "nnUNetV2Runner",
            "input_config": "$@data_src_cfg",
            "trainer_class_name": "@nnunet_trainer_class_name",
            "work_dir": "@nnunet_root_folder",
        }

        train_config["network"] = "$@nnunet_trainer.network._orig_mod"

        train_handlers = train_config["train_handlers"]["handlers"]

        for idx, handler in enumerate(train_handlers):
            if handler["_target_"] == "ValidationHandler":
                train_handlers.pop(idx)
                break

        train_config["train_handlers"]["handlers"] = train_handlers

        if train_extra_configs is not None and "resume_epoch" in train_extra_configs:
            resume_epoch = train_extra_configs["resume_epoch"]
            train_config["initialize"] = [
                "$monai.utils.set_determinism(seed=123)",
                "$@runner.dataset_name_or_id",
                f"$src.trainer.reload_checkpoint(@train#trainer, {resume_epoch}, @iterations, @ckpt_dir, @lr_scheduler)",
            ]
        else:
            train_config["initialize"] = ["$monai.utils.set_determinism(seed=123)", "$@runner.dataset_name_or_id"]

        if "Val_Dice" in train_config["val_key_metric"]:
            train_config["val_key_metric"] = {"Val_Dice_Local": train_config["val_key_metric"]["Val_Dice"]}

        if "Val_Dice_per_class" in train_config["val_additional_metrics"]:
            train_config["val_additional_metrics"] = {
                "Val_Dice_per_class_Local": train_config["val_additional_metrics"]["Val_Dice_per_class"]
            }
        if "nnunet_plans_identifier" in bundle_config:
            train_config["nnunet_plans_identifier"] = bundle_config["nnunet_plans_identifier"]

        if "nnunet_trainer_class_name" in bundle_config:
            train_config["nnunet_trainer_class_name"] = bundle_config["nnunet_trainer_class_name"]

        if train_extra_configs is not None:
            for key in train_extra_configs:
                train_config[key] = train_extra_configs[key]

    with open(Path(bundle_config["bundle_root"]).joinpath("configs", "train.json"), "w") as f:
        json.dump(train_config, f)

    with open(Path(bundle_config["bundle_root"]).joinpath("configs", "train.yaml"), "w") as f:
        yaml.dump(train_config, f)

    if not Path(bundle_config["bundle_root"]).joinpath("configs", "evaluate.yaml").exists():
        shutil.copy(
            Path(bundle_config["bundle_root"]).joinpath("nnUNet", "evaluator", "evaluator.yaml"),
            Path(bundle_config["bundle_root"]).joinpath("configs", "evaluate.yaml"),
        )

    with open(Path(bundle_config["bundle_root"]).joinpath("configs", "evaluate.yaml")) as f:
        evaluate_config = yaml.safe_load(f)
        evaluate_config["bundle_root"] = bundle_config["bundle_root"]

        evaluate_config["tracking_uri"] = bundle_config["tracking_uri"]
        evaluate_config["mlflow_experiment_name"] = bundle_config["mlflow_experiment_name"]
        evaluate_config["mlflow_run_name"] = bundle_config["mlflow_run_name"]

        if "nnunet_plans_identifier" in bundle_config:
            evaluate_config["nnunet_plans_identifier"] = bundle_config["nnunet_plans_identifier"]
        if "nnunet_trainer_class_name" in bundle_config:
            evaluate_config["nnunet_trainer_class_name"] = bundle_config["nnunet_trainer_class_name"]

    with open(Path(bundle_config["bundle_root"]).joinpath("configs", "evaluate.json"), "w") as f:
        json.dump(evaluate_config, f)

    with open(Path(bundle_config["bundle_root"]).joinpath("configs", "evaluate.yaml"), "w") as f:
        yaml.dump(evaluate_config, f)


def finalize_bundle(bundle_root, nnunet_root_dir=None, validate_with_nnunet=True,
                    experiment_name=None, client_name=None, tracking_uri=None,
                    dataset_name_or_id=None, trainer_class_name="nnUNetTrainer",
                    nnunet_plans_name="nnUNetPlans", fold=0, mlflow_token=None):
    """
    Finalizes a MONAI bundle by converting model and dataset configurations to nnUNet format,
    saving checkpoints, and optionally validating the model using nnUNet.
    
    Parameters
    ----------
    bundle_root : str
        Path to the root directory of the MONAI bundle.
    nnunet_root_dir : str, optional
        Path to the nnUNet root directory. Required if `validate_with_nnunet` is True.
    validate_with_nnunet : bool, optional
        Whether to validate the model using nnUNet. Default is True.
    experiment_name : str, optional
        Name of the MLflow experiment for logging validation metrics.
    client_name : str, optional
        Name of the client for tagging MLflow runs.
    tracking_uri : str, optional
        URI of the MLflow tracking server.
    dataset_name_or_id : str, optional
        Name or ID of the dataset for nnUNet validation.
    trainer_class_name : str, optional
        Name of the nnUNet trainer class. Default is "nnUNetTrainer".
    nnunet_plans_name : str, optional
        Name of the nnUNet plans. Default is "nnUNetPlans".
    fold : int, optional
        Fold number for nnUNet training and validation. Default is 0.
    mlflow_token : str, optional
        Token for authenticating with the MLflow tracking server.
    
    Returns
    -------
    dict
        A dictionary containing the validation summary metrics if `validate_with_nnunet` is True.
        Otherwise, returns None.
    
    Notes
    -----
    - This function assumes the MONAI bundle contains `plans.json` and `dataset.json` files
        in the `models` directory.
    - If `validate_with_nnunet` is True, the function converts the MONAI bundle to nnUNet format,
        trains a single model, and logs validation metrics to MLflow.
    - The function creates and saves nnUNet-compatible checkpoints in the `models` directory.
    """
    print("Finalizing bundle...")
    if nnunet_root_dir is None:
        raise ValueError("nnunet_root_dir must be provided if validate_with_nnunet is True")
    if not Path(bundle_root).joinpath("models", "plans.json").exists():
        raise ValueError("plans.json file not found in the models directory of the bundle")
    if not Path(bundle_root).joinpath("models", "dataset.json").exists():
        raise ValueError("dataset.json file not found in the models directory of the bundle")
    
    print("Converting bundle to nnUNet format...")
    
    with open(Path(bundle_root).joinpath("models","plans.json"),"r") as f:
        plans = json.load(f)

    with open(Path(bundle_root).joinpath("configs","plans.yaml"),"w") as f:
        yaml.dump({"plans": plans}, f)

    with open(Path(bundle_root).joinpath("models","dataset.json"),"r") as f:
        dataset_json = json.load(f)
        
    with open(Path(bundle_root).joinpath("configs","dataset.yaml"),"w") as f:
        yaml.dump({"dataset_json": dataset_json}, f)
        
    checkpoint = {
        "trainer_name": trainer_class_name,
        "inference_allowed_mirroring_axes": (0, 1, 2),
        "init_args": {
            "configuration": "3d_fullres",
    }
    }
    
    torch.save(checkpoint, Path(bundle_root).joinpath("models","nnunet_checkpoint.pth"))
    
    checkpoint_dict = torch.load(Path(bundle_root).joinpath("models",f"fold_{fold}","FL_global_model.pt"))
    
    new_checkpoint_dict = {}
    new_checkpoint_dict["network_weights"] = checkpoint_dict["model"]
    torch.save(new_checkpoint_dict, Path(bundle_root).joinpath("models",f"fold_{fold}","checkpoint_epoch=1000.pt"))
    torch.save(new_checkpoint_dict, Path(bundle_root).joinpath("models",f"fold_{fold}","checkpoint_key_metric=1.0.pt"))
    
    if validate_with_nnunet:
        data_src_cfg = os.path.join(nnunet_root_dir, f"Task{dataset_name_or_id}_data_src_cfg.yaml")
        runner = nnUNetV2Runner(input_config=data_src_cfg, trainer_class_name=trainer_class_name, work_dir=nnunet_root_dir)

        nnunet_config = {"dataset_name_or_id": dataset_name_or_id, "nnunet_trainer": trainer_class_name}
        convert_monai_bundle_to_nnunet(nnunet_config, bundle_root)
        
        runner.train_single_model(config="3d_fullres", fold=fold, val="")
        
        if mlflow_token is not None:
            os.environ["MLFLOW_TRACKING_TOKEN"] = mlflow_token
        if tracking_uri is not None:
            mlflow.set_tracking_uri(tracking_uri)

        try:
            mlflow.create_experiment(experiment_name)
        except Exception as e:
            print(e)
            mlflow.set_experiment(experiment_id=(mlflow.get_experiment_by_name(experiment_name).experiment_id))

        filter = f"""
        tags."client" = "{client_name}"
        """

        runs = mlflow.search_runs(experiment_names=[experiment_name], filter_string=filter, order_by=["start_time DESC"])

        validation_summary = os.path.join(
            runner.nnunet_results,
            runner.dataset_name,
            f"{trainer_class_name}__{nnunet_plans_name}__3d_fullres",
            f"fold_{fold}",
            "validation",
            "summary.json",
        )

        dataset_file = os.path.join(
            runner.nnunet_results,
            runner.dataset_name,
            f"{trainer_class_name}__{nnunet_plans_name}__3d_fullres",
            "dataset.json",
        )

        with open(dataset_file, "r") as f:
            dataset_dict = json.load(f)
            labels = dataset_dict["labels"]
            labels = {str(v): k for k, v in labels.items()}

        with open(validation_summary, "r") as f:
            validation_summary_dict = json.load(f)

        if len(runs) == 0:
            with mlflow.start_run(run_name=f"run_{client_name}", tags={"client": client_name}):
                for label in validation_summary_dict["mean"]:
                    for metric in validation_summary_dict["mean"][label]:
                        label_name = labels[label]
                        mlflow.log_metric(f"{label_name}_{metric}", float(validation_summary_dict["mean"][label][metric]))

        else:
            with mlflow.start_run(run_id=runs.iloc[0].run_id, tags={"client": client_name}):
                for label in validation_summary_dict["mean"]:
                    for metric in validation_summary_dict["mean"][label]:
                        label_name = labels[label]
                        mlflow.log_metric(f"{label_name}_{metric}", float(validation_summary_dict["mean"][label][metric]))

        return validation_summary_dict
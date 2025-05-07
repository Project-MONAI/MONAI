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
from pyhocon import ConfigFactory
from pyhocon.converter import HOCONConverter

from monai.nvflare.utils import prepare_data_folder_api, finalize_bundle_api, cross_site_evaluation_api, plan_and_preprocess_api, prepare_bundle_api, train_api, validation_api





def run_job(sess, task_name, job_folder, clients=None):
    """
    Submits a job to the session with the specified task name and job folder. Optionally, 
    generates a meta configuration file for the job if clients are provided.
    
    Parameters
    ----------
    sess : object
        The session object used to submit the job.
    task_name : str
        The name of the task to be executed.
    job_folder : str or Path
        The path to the folder where the job configuration and related files are stored.
    clients : dict, optional
        A dictionary of client IDs as keys and their respective configurations as values. 
        If provided, a meta configuration file is generated for the job.
    
    Raises
    ------
    FileNotFoundError
        If the specified job folder or task folder does not exist.
    IOError
        If there is an error writing the meta configuration file.
    
    Notes
    -----
    - The `meta.conf` file is created in the task folder within the job folder if `clients` is provided.
    - The `sess.submit_job` method is called to submit the job to the session.
    """
    if clients is not None:
        meta = {
            "name": f"{task_name}_nnUNet",
            "resource_spec": {},
            "deploy_map": {f"{task_name}-server": ["server"]},
            "min_clients": 1,
            "mandatory_clients": list(clients.keys()),
        }
        for client_id in clients:
            meta["deploy_map"][f"{task_name}-client-{client_id}"] = [client_id]

        with open(Path(job_folder).joinpath(task_name).joinpath("meta.conf"), "w") as f:
            f.write("{\n")
            f.write(HOCONConverter.to_hocon(ConfigFactory.from_dict(meta)))
            f.write("\n}")
            
    job_id = sess.submit_job(str(Path(job_folder).joinpath(task_name)))
    
    return job_id


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
    skip_training=False,
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
    
    train_api(nnunet_root_dir, dataset_name_or_id, experiment_name, trainer_class_name, run_with_bundle, bundle_root, skip_training, continue_training, fold, tracking_uri, client_name, resume_epoch)

    validation_summary_dict, labels = validation_api(nnunet_root_dir, dataset_name_or_id, trainer_class_name, nnunet_plans_name, fold)
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
            mlflow.log_dict(validation_summary_dict, "validation_summary.json")
            for label in validation_summary_dict["mean"]:
                for metric in validation_summary_dict["mean"][label]:
                    label_name = labels[label]
                    mlflow.log_metric(f"{label_name}_{metric}", float(validation_summary_dict["mean"][label][metric]))

    else:
        with mlflow.start_run(run_id=runs.iloc[0].run_id, tags={"client": client_name}):
            mlflow.log_dict(validation_summary_dict, "validation_summary.json")
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

    return {"original_dataset_name": str(nnunet_plans["original_dataset_name"]), "dataset_name": str(nnunet_plans["dataset_name"])}


def plan_and_preprocess(
    nnunet_root_dir,
    dataset_name_or_id,
    client_name,
    experiment_name,
    tracking_uri,
    mlflow_token=None,
    nnunet_plans_name="nnUNetPlans",
    trainer_class_name="nnUNetTrainer",
    dataset_name=None,
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

    nnunet_plans = plan_and_preprocess_api(
        nnunet_root_dir=nnunet_root_dir,
        dataset_name_or_id=dataset_name_or_id,
        trainer_class_name=trainer_class_name,
        nnunet_plans_name=nnunet_plans_name,
    )

    if mlflow_token is not None:
        os.environ["MLFLOW_TRACKING_TOKEN"] = mlflow_token
    if tracking_uri is not None:
        mlflow.set_tracking_uri(tracking_uri)

    try:
        mlflow.create_experiment(experiment_name)
    except Exception as e:
        print(e)
        mlflow.set_experiment(experiment_id=(mlflow.get_experiment_by_name(experiment_name).experiment_id))

    run_name = f"run_plan_and_preprocess_{client_name}"

    runs = mlflow.search_runs(
    experiment_names=[experiment_name],
    filter_string=f"tags.mlflow.runName = '{run_name}'",
    order_by=["start_time DESC"]
    )
    tags = {"client": client_name}
    if dataset_name is not None:
        tags["dataset_name"] = dataset_name

    if len(runs) == 0:
        with mlflow.start_run(run_name=f"run_plan_and_preprocess_{client_name}", tags=tags):
            mlflow.log_dict(nnunet_plans, nnunet_plans_name + ".json")

    else:
        with mlflow.start_run(run_id=runs.iloc[0].run_id, tags=tags):
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
    dataset_name=None,
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

    data_list = prepare_data_folder_api(
        data_dir,
        nnunet_root_dir,
        dataset_name_or_id,
        modality_dict,
        experiment_name,
        dataset_format=dataset_format,
        modality_list=modality_list,
        subfolder_suffix=subfolder_suffix,
        patient_id_in_file_identifier=patient_id_in_file_identifier,
        trainer_class_name=trainer_class_name,
    )

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


    run_name = f"run_prepare_{client_name}"

    runs = mlflow.search_runs(
    experiment_names=[experiment_name],
    filter_string=f"tags.mlflow.runName = '{run_name}'",
    order_by=["start_time DESC"]
    )
    tags = {"client": client_name}
    if dataset_name is not None:
        tags["dataset_name"] = dataset_name
    try:
        if len(runs) == 0:
            with mlflow.start_run(run_name=f"run_prepare_{client_name}", tags=tags):
                mlflow.log_table(pd.DataFrame.from_records(data_list["training"]), f"{client_name}_train.json")
        else:
            with mlflow.start_run(run_id=runs.iloc[0].run_id, tags=tags):
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

    prepare_bundle_api(bundle_config, train_extra_configs=train_extra_configs, is_federated=True)




def finalize_bundle(bundle_root, nnunet_root_dir=None, validate_with_nnunet=True,
                    experiment_name=None, client_name=None, tracking_uri=None,
                    dataset_name_or_id=None, trainer_class_name="nnUNetTrainer",
                    nnunet_plans_name="nnUNetPlans", fold=0, mlflow_token=None, dataset_name=None):
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
    finalize_bundle_api(nnunet_root_dir, bundle_root, trainer_class_name, fold)
   
    
    if validate_with_nnunet:
        nnunet_config = {"dataset_name_or_id": dataset_name_or_id, "nnunet_trainer": trainer_class_name}
        convert_monai_bundle_to_nnunet(nnunet_config, bundle_root)
        validation_summary_dict, labels = validation_api(nnunet_root_dir, dataset_name_or_id, trainer_class_name, nnunet_plans_name, fold)
        
        
        if mlflow_token is not None:
            os.environ["MLFLOW_TRACKING_TOKEN"] = mlflow_token
        if tracking_uri is not None:
            mlflow.set_tracking_uri(tracking_uri)

        try:
            mlflow.create_experiment("FedLearning-"+experiment_name)
        except Exception as e:
            print(e)
            mlflow.set_experiment(experiment_id=(mlflow.get_experiment_by_name("FedLearning-"+experiment_name).experiment_id))

        run_name = f"run_validation_{client_name}"

        runs = mlflow.search_runs(
        experiment_names=["FedLearning-"+experiment_name],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
        order_by=["start_time DESC"]
        )
        tags = {"client": client_name}
        if dataset_name is not None:
            tags["dataset_name"] = dataset_name


        if len(runs) == 0:
            with mlflow.start_run(run_name=f"run_validation_{client_name}", tags=tags):
                mlflow.log_dict(validation_summary_dict, "validation_summary.json")
                for label in validation_summary_dict["mean"]:
                    for metric in validation_summary_dict["mean"][label]:
                        label_name = labels[label]
                        mlflow.log_metric(f"{label_name}_{metric}", float(validation_summary_dict["mean"][label][metric]))

        else:
            with mlflow.start_run(run_id=runs.iloc[0].run_id, tags=tags):
                mlflow.log_dict(validation_summary_dict, "validation_summary.json")
                for label in validation_summary_dict["mean"]:
                    for metric in validation_summary_dict["mean"][label]:
                        label_name = labels[label]
                        mlflow.log_metric(f"{label_name}_{metric}", float(validation_summary_dict["mean"][label][metric]))

        return validation_summary_dict


def run_cross_site_validation(nnunet_root_dir, dataset_name_or_id, app_path, app_model_path, app_output_path, trainer_class_name="nnUNetTrainer", fold=0,
                    experiment_name=None, client_name=None, tracking_uri=None,
                    nnunet_plans_name="nnUNetPlans", mlflow_token=None, skip_prediction=False, dataset_name=None):

    validation_summary_dict, labels = cross_site_evaluation_api(
        nnunet_root_dir,
        dataset_name_or_id,
        app_path,
        app_model_path,
        app_output_path,
        trainer_class_name=trainer_class_name,
        fold=fold,
        nnunet_plans_name=nnunet_plans_name,
        skip_prediction=skip_prediction,
    )
    if mlflow_token is not None:
        os.environ["MLFLOW_TRACKING_TOKEN"] = mlflow_token
    if tracking_uri is not None:
        mlflow.set_tracking_uri(tracking_uri)

    try:
        mlflow.create_experiment(experiment_name)
    except Exception as e:
        print(e)
        mlflow.set_experiment(experiment_id=(mlflow.get_experiment_by_name(experiment_name).experiment_id))

    run_name = f"run_cross_site_validation_{client_name}"

    runs = mlflow.search_runs(
    experiment_names=[experiment_name],
    filter_string=f"tags.mlflow.runName = '{run_name}'",
    order_by=["start_time DESC"]
    )
    tags = {"client": client_name}
    if dataset_name is not None:
        tags["dataset_name"] = dataset_name


    if len(runs) == 0:
        with mlflow.start_run(run_name=f"run_{client_name}", tags={"client": client_name}):
            mlflow.log_dict(validation_summary_dict, "validation_summary.json")
            for label in validation_summary_dict["mean"]:
                for metric in validation_summary_dict["mean"][label]:
                    label_name = labels[label]
                    mlflow.log_metric(f"{label_name}_{metric}", float(validation_summary_dict["mean"][label][metric]))

    else:
        with mlflow.start_run(run_id=runs.iloc[0].run_id, tags={"client": client_name}):
            mlflow.log_dict(validation_summary_dict, "validation_summary.json")
            for label in validation_summary_dict["mean"]:
                for metric in validation_summary_dict["mean"][label]:
                    label_name = labels[label]
                    mlflow.log_metric(f"{label_name}_{metric}", float(validation_summary_dict["mean"][label][metric]))

    return validation_summary_dict
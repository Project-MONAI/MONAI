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
import sys
from pathlib import Path

from nvflare.apis.dxo import DXO, DataKind
from nvflare.apis.event_type import EventType
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal

from monai.nvflare.nvflare_nnunet import (  # check_host_config,
    check_packages,
    plan_and_preprocess,
    prepare_bundle,
    prepare_data_folder,
    preprocess,
    train,
    finalize_bundle,
    run_cross_site_validation
)


class nnUNetExecutor(Executor):
    """
    nnUNetExecutor is a class that handles the execution of various tasks related to nnUNet training and preprocessing
    within the NVFlare framework.

    Parameters
    ----------
    data_dir : str, optional
        Directory where the data is stored.
    modality_dict : dict, optional
        Dictionary containing modality information.
    prepare_task_name : str, optional
        Name of the task for preparing the dataset.
    check_client_packages_task_name : str, optional
        Name of the task for checking client packages.
    plan_and_preprocess_task_name : str, optional
        Name of the task for planning and preprocessing.
    preprocess_task_name : str, optional
        Name of the task for preprocessing.
    training_task_name : str, optional
        Name of the task for training.
    prepare_bundle_name : str, optional
        Name of the task for preparing the bundle.
    subfolder_suffix : str, optional
        Suffix for subfolders.
    dataset_format : str, optional
        Format of the dataset, default is "subfolders".
    patient_id_in_file_identifier : bool, optional
        Whether patient ID is in file identifier, default is True.
    nnunet_config : dict, optional
        Configuration dictionary for nnUNet.
    nnunet_root_folder : str, optional
        Root folder for nnUNet.
    client_name : str, optional
        Name of the client.
    tracking_uri : str, optional
        URI for tracking.
    mlflow_token : str, optional
        Token for MLflow.
    bundle_root : str, optional
        Root directory for the bundle.
    train_extra_configs : dict, optional
        Extra configurations for training.
    exclude_vars : list, optional
        List of variables to exclude.
    modality_list : list, optional
        List of modalities.

    Methods
    -------
    handle_event(event_type: str, fl_ctx: FLContext)
        Handles events triggered during the federated learning process.
    initialize(fl_ctx: FLContext)
        Initializes the executor with the given federated learning context.
    execute(task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable
        Executes the specified task.
    prepare_dataset() -> Shareable
        Prepares the dataset for training.
    check_packages_installed() -> Shareable
        Checks if the required packages are installed.
    plan_and_preprocess() -> Shareable
        Plans and preprocesses the dataset.
    preprocess() -> Shareable
        Preprocesses the dataset.
    train() -> Shareable
        Trains the model.
    prepare_bundle() -> Shareable
        Prepares the bundle for deployment.
    """

    def __init__(
        self,
        data_dir=None,
        modality_dict=None,
        prepare_task_name="prepare",
        check_client_packages_task_name="check_client_packages",
        plan_and_preprocess_task_name="plan_and_preprocess",
        preprocess_task_name="preprocess",
        training_task_name="train",
        finalize_task_name="finalize",
        prepare_bundle_name="prepare_bundle",
        cross_site_validation_task_name="cross_site_validation",
        subfolder_suffix=None,
        dataset_format="subfolders",
        patient_id_in_file_identifier=True,
        nnunet_config=None,
        nnunet_root_folder=None,
        client_name=None,
        tracking_uri=None,
        mlflow_token=None,
        bundle_root=None,
        modality_list=None,
        monai_deploy_config=None,
        train_extra_configs=None,
        exclude_vars=None,
        continue_training=False,
        label_dict = None
    ):
        super().__init__()

        self.exclude_vars = exclude_vars
        self.prepare_task_name = prepare_task_name
        self.data_dir = data_dir
        self.subfolder_suffix = subfolder_suffix
        self.patient_id_in_file_identifier = patient_id_in_file_identifier
        self.dataset_format = dataset_format
        self.modality_dict = modality_dict
        self.nnunet_config = nnunet_config
        self.nnunet_root_folder = nnunet_root_folder
        self.client_name = client_name
        self.tracking_uri = tracking_uri
        self.mlflow_token = mlflow_token
        self.check_client_packages_task_name = check_client_packages_task_name
        self.plan_and_preprocess_task_name = plan_and_preprocess_task_name
        self.preprocess_task_name = preprocess_task_name
        self.training_task_name = training_task_name
        self.prepare_bundle_name = prepare_bundle_name
        self.bundle_root = bundle_root
        self.train_extra_configs = train_extra_configs
        self.modality_list = modality_list
        self.continue_training = continue_training
        self.finalize_task_name = finalize_task_name
        self.cross_site_validation_task_name = cross_site_validation_task_name
        self.monai_deploy_config = monai_deploy_config
        self.label_dict = label_dict

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            self.initialize(fl_ctx)

    def initialize(self, fl_ctx: FLContext):
        self.run_dir = fl_ctx.get_engine().get_workspace().get_run_dir(fl_ctx.get_job_id())
        self.root_dir = fl_ctx.get_engine().get_workspace().root_dir
        self.custom_app_dir = fl_ctx.get_engine().get_workspace().get_app_custom_dir(fl_ctx.get_job_id())

        with open("init_logfile_out.log", "w") as f_o:
            with open("init_logfile_err.log", "w") as f_e:
                subprocess.call(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        "--user",
                        "-r",
                        str(Path(self.custom_app_dir).joinpath("requirements.txt")),
                    ],
                    stdout=f_o,
                    stderr=f_e,
                )

    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        self.run_dir = fl_ctx.get_engine().get_workspace().get_run_dir(fl_ctx.get_job_id())
        self.root_dir = fl_ctx.get_engine().get_workspace().root_dir
        self.custom_app_dir = fl_ctx.get_engine().get_workspace().get_app_custom_dir(fl_ctx.get_job_id())
        try:
            if task_name == self.prepare_task_name:
                return self.prepare_dataset()
            elif task_name == self.check_client_packages_task_name:
                return self.check_packages_installed()
            elif task_name == self.plan_and_preprocess_task_name:
                return self.plan_and_preprocess()
            elif task_name == self.preprocess_task_name:
                return self.preprocess()
            elif task_name == self.training_task_name:
                return self.train()
            elif task_name == self.prepare_bundle_name:
                return self.prepare_bundle()
            elif task_name == self.finalize_task_name:
                return self.finalize_bundle()
            elif task_name == self.cross_site_validation_task_name:
                return self.run_cross_site_validation()
            else:
                return make_reply(ReturnCode.TASK_UNKNOWN)
        except Exception as e:
            self.log_exception(fl_ctx, f"Exception in simple trainer: {e}.")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

    def prepare_dataset(self) -> Shareable:
        if "nnunet_trainer" not in self.nnunet_config:
            nnunet_trainer_name = "nnUNetTrainer"
        else:
            nnunet_trainer_name = self.nnunet_config["nnunet_trainer"]

        data_list = prepare_data_folder(
            data_dir=self.data_dir,
            nnunet_root_dir=self.nnunet_root_folder,
            dataset_name_or_id=self.nnunet_config["dataset_name_or_id"],
            modality_dict=self.modality_dict,
            experiment_name=self.nnunet_config["experiment_name"],
            client_name=self.client_name,
            dataset_format=self.dataset_format,
            patient_id_in_file_identifier=self.patient_id_in_file_identifier,
            tracking_uri=self.tracking_uri,
            mlflow_token=self.mlflow_token,
            subfolder_suffix=self.subfolder_suffix,
            trainer_class_name=nnunet_trainer_name,
            modality_list=self.modality_list,
        )

        outgoing_dxo = DXO(data_kind=DataKind.COLLECTION, data=data_list, meta={})
        return outgoing_dxo.to_shareable()

    def check_packages_installed(self):
        packages = [
            "nvflare",
            # {"package_name":'pymaia-learn',"import_name":"PyMAIA"},
            "torch",
            "monai",
            "numpy",
            "nnunetv2",
        ]
        package_report = check_packages(packages)

        # host_config = check_host_config()
        # package_report.update(host_config)

        outgoing_dxo = DXO(data_kind=DataKind.COLLECTION, data=package_report, meta={})

        return outgoing_dxo.to_shareable()

    def plan_and_preprocess(self):
        if "nnunet_plans" not in self.nnunet_config:
            nnunet_plans_name = "nnUNetPlans"
        else:
            nnunet_plans_name = self.nnunet_config["nnunet_plans"]

        if "nnunet_trainer" not in self.nnunet_config:
            nnunet_trainer_name = "nnUNetTrainer"
        else:
            nnunet_trainer_name = self.nnunet_config["nnunet_trainer"]

        nnunet_plans = plan_and_preprocess(
            self.nnunet_root_folder,
            self.nnunet_config["dataset_name_or_id"],
            self.client_name,
            self.nnunet_config["experiment_name"],
            self.tracking_uri,
            nnunet_plans_name=nnunet_plans_name,
            trainer_class_name=nnunet_trainer_name,
        )

        outgoing_dxo = DXO(data_kind=DataKind.COLLECTION, data=nnunet_plans, meta={})
        return outgoing_dxo.to_shareable()

    def preprocess(self):
        if "nnunet_plans" not in self.nnunet_config:
            nnunet_plans_name = "nnUNetPlans"
        else:
            nnunet_plans_name = self.nnunet_config["nnunet_plans"]

        if "nnunet_trainer" not in self.nnunet_config:
            nnunet_trainer_name = "nnUNetTrainer"
        else:
            nnunet_trainer_name = self.nnunet_config["nnunet_trainer"]

        nnunet_plans = preprocess(
            self.nnunet_root_folder,
            self.nnunet_config["dataset_name_or_id"],
            nnunet_plans_file_path=Path(self.custom_app_dir).joinpath(f"{nnunet_plans_name}.json"),
            trainer_class_name=nnunet_trainer_name,
        )
        outgoing_dxo = DXO(data_kind=DataKind.COLLECTION, data=nnunet_plans, meta={})
        return outgoing_dxo.to_shareable()

    def train(self):
        if "nnunet_trainer" not in self.nnunet_config:
            nnunet_trainer_name = "nnUNetTrainer"
        else:
            nnunet_trainer_name = self.nnunet_config["nnunet_trainer"]

        if "nnunet_plans" not in self.nnunet_config:
            nnunet_plans_name = "nnUNetPlans"
        else:
            nnunet_plans_name = self.nnunet_config["nnunet_plans"]

        validation_summary = train(
            self.nnunet_root_folder,
            trainer_class_name=nnunet_trainer_name,
            fold=0,
            experiment_name=self.nnunet_config["experiment_name"],
            client_name=self.client_name,
            tracking_uri=self.tracking_uri,
            nnunet_plans_name=nnunet_plans_name,
            dataset_name_or_id=self.nnunet_config["dataset_name_or_id"],
            run_with_bundle=True if self.bundle_root is not None else False,
            bundle_root=self.bundle_root,
            continue_training=self.continue_training
        )
        outgoing_dxo = DXO(data_kind=DataKind.COLLECTION, data=validation_summary, meta={})
        return outgoing_dxo.to_shareable()

    def prepare_bundle(self):
        if "nnunet_trainer" not in self.nnunet_config:
            nnunet_trainer_name = "nnUNetTrainer"
        else:
            nnunet_trainer_name = self.nnunet_config["nnunet_trainer"]

        if "nnunet_plans" not in self.nnunet_config:
            nnunet_plans_name = "nnUNetPlans"
        else:
            nnunet_plans_name = self.nnunet_config["nnunet_plans"]

        bundle_config = {
            "bundle_root": self.bundle_root,
            "tracking_uri": self.tracking_uri,
            "mlflow_experiment_name": "FedLearning-" + self.nnunet_config["experiment_name"],
            "mlflow_run_name": self.client_name,
            "nnunet_plans_identifier": nnunet_plans_name,
            "nnunet_trainer_class_name": nnunet_trainer_name,
            "dataset_name_or_id": self.nnunet_config["dataset_name_or_id"],
            "label_dict": self.label_dict,
        }

        bundle_config = prepare_bundle(bundle_config, self.train_extra_configs)

        outgoing_dxo = DXO(data_kind=DataKind.COLLECTION, data=bundle_config, meta={})
        
        return outgoing_dxo.to_shareable()

    def finalize_bundle(self):
        
        if "nnunet_trainer" not in self.nnunet_config:
            nnunet_trainer_name = "nnUNetTrainer"
        else:
            nnunet_trainer_name = self.nnunet_config["nnunet_trainer"]

        if "nnunet_plans" not in self.nnunet_config:
            nnunet_plans_name = "nnUNetPlans"
        else:
            nnunet_plans_name = self.nnunet_config["nnunet_plans"]

        validation_summary = finalize_bundle(
            self.bundle_root,
            self.nnunet_root_folder,
            trainer_class_name=nnunet_trainer_name,
            fold=0,
            experiment_name=self.nnunet_config["experiment_name"],
            client_name=self.client_name,
            tracking_uri=self.tracking_uri,
            nnunet_plans_name=nnunet_plans_name,
            dataset_name_or_id=self.nnunet_config["dataset_name_or_id"]
        )
        outgoing_dxo = DXO(data_kind=DataKind.COLLECTION, data=validation_summary, meta={})
        return outgoing_dxo.to_shareable()
        
    def run_cross_site_validation(self):
        if "nnunet_trainer" not in self.nnunet_config:
            nnunet_trainer_name = "nnUNetTrainer"
        else:
            nnunet_trainer_name = self.nnunet_config["nnunet_trainer"]

        if "nnunet_plans" not in self.nnunet_config:
            nnunet_plans_name = "nnUNetPlans"
        else:
            nnunet_plans_name = self.nnunet_config["nnunet_plans"]
        
        validation_summary = run_cross_site_validation(
            self.nnunet_root_folder,
            self.nnunet_config["dataset_name_or_id"],
            self.monai_deploy_config["app_path"],
            self.monai_deploy_config["app_model_path"],
            self.monai_deploy_config["app_output_path"],
            trainer_class_name=nnunet_trainer_name,
            fold=0,
            experiment_name=self.nnunet_config["experiment_name"],
            client_name=self.client_name,
            tracking_uri=self.tracking_uri,
            nnunet_plans_name=nnunet_plans_name
        )
        
        outgoing_dxo = DXO(data_kind=DataKind.COLLECTION, data=validation_summary, meta={})
        return outgoing_dxo.to_shareable()
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

# pylint: disable=import-error
from __future__ import annotations

import glob
import os
import subprocess
from typing import Any

import monai
from monai.apps.nnunet.utils import NNUNETMode as M  # noqa: N814
from monai.apps.nnunet.utils import analyze_data, create_new_data_copy, create_new_dataset_json
from monai.bundle import ConfigParser
from monai.utils import ensure_tuple, optional_import

load_pickle, _ = optional_import("batchgenerators.utilities.file_and_folder_operations", name="load_pickle")
join, _ = optional_import("batchgenerators.utilities.file_and_folder_operations", name="join")
tqdm, has_tqdm = optional_import("tqdm", name="tqdm")
nib, _ = optional_import("nibabel")

logger = monai.apps.utils.get_logger(__name__)

__all__ = ["nnUNetV2Runner"]


class nnUNetV2Runner:  # noqa: N801
    """
    ``nnUNetV2Runner`` provides an interface in MONAI to use `nnU-Net` V2 library to analyze, train, and evaluate
    neural networks for medical image segmentation tasks.
    A version of nnunetv2 higher than 2.2 is needed for this class.

    ``nnUNetV2Runner`` can be used in two ways:

    #. with one line of code to execute the complete pipeline.
    #. with a series of commands to run each modules in the pipeline.

    The output of the interface is a directory that contains:

    #. converted dataset met the requirement of nnU-Net V2
    #. data analysis results
    #. checkpoints from the trained U-Net models
    #. validation accuracy in each fold of cross-validation
    #. the predictions on the testing datasets from the final algorithm ensemble and potential post-processing

    Args:
        input_config: the configuration dictionary or the file path to the configuration in the form of YAML.
            The keys required in the configuration are:
            - ``"datalist"``: File path to the datalist for the train/testing splits
            - ``"dataroot"``: File path to the dataset
            - ``"modality"``: Imaging modality, e.g. "CT", ["T2", "ADC"]
            Currently, the configuration supports these optional keys:
            - ``"nnunet_raw"``: File path that will be written to env variable for nnU-Net
            - ``"nnunet_preprocessed"``: File path that will be written to env variable for nnU-Net
            - ``"nnunet_results"``: File path that will be written to env variable for nnU-Net
            - ``"nnUNet_trained_models"``
            - ``"dataset_name_or_id"``: Name or Integer ID of the dataset
            If an optional key is not specified, then the pipeline will use the default values.
        trainer_class_name: the trainer class names offered by nnUNetV2 exhibit variations in training duration.
            Default: "nnUNetTrainer". Other options: "nnUNetTrainer_Xepoch". X could be one of 1,5,10,20,50,100,
            250,2000,4000,8000.
        export_validation_probabilities: True to save softmax predictions from final validation as npz
            files (in addition to predicted segmentations). Needed for finding the best ensemble.
            Default: True.
        work_dir: working directory to save the intermediate and final results.

    Examples:
        - Use the one-liner to start the nnU-Net workflow

        .. code-block:: bash

            python -m monai.apps.nnunet nnUNetV2Runner run --input_config ./input.yaml

        - Use `convert_dataset` to prepare the data to meet nnU-Net requirements, generate dataset JSON file,
            and copy the dataset to a location specified by ``nnunet_raw`` in the input config file

        .. code-block:: bash

            python -m monai.apps.nnunet nnUNetV2Runner convert_dataset --input_config="./input.yaml"

        - `convert_msd_dataset` is an alternative option to prepare the data if the dataset is MSD.

        .. code-block:: bash

            python -m monai.apps.nnunet nnUNetV2Runner convert_msd_dataset \\
                --input_config "./input.yaml" --data_dir "/path/to/Task09_Spleen"

        - experiment planning and data pre-processing

        .. code-block:: bash

            python -m monai.apps.nnunet nnUNetV2Runner plan_and_process --input_config "./input.yaml"

        - training all 20 models using all GPUs available.
            "CUDA_VISIBLE_DEVICES" environment variable is not supported.

        .. code-block:: bash

            python -m monai.apps.nnunet nnUNetV2Runner train --input_config "./input.yaml"

        - training a single model on a single GPU for 5 epochs. Here ``config`` is used to specify the configuration.

        .. code-block:: bash

            python -m monai.apps.nnunet nnUNetV2Runner train_single_model --input_config "./input.yaml" \\
                --config "3d_fullres" \\
                --fold 0 \\
                --gpu_id 0 \\
                --trainer_class_name "nnUNetTrainer_5epochs" \\
                --export_validation_probabilities True

        - training for all 20 models (4 configurations by 5 folds) on 2 GPUs

        .. code-block:: bash

            python -m monai.apps.nnunet nnUNetV2Runner train --input_config "./input.yaml" --gpu_id_for_all "0,1"

        - 5-fold training for a single model on 2 GPUs. Here ``configs`` is used to specify the configurations.

        .. code-block:: bash

            python -m monai.apps.nnunet nnUNetV2Runner train --input_config "./input.yaml" \\
                --configs "3d_fullres" \\
                --trainer_class_name "nnUNetTrainer_5epochs" \\
                --export_validation_probabilities True \\
                --gpu_id_for_all "0,1"

        - find the best configuration

        .. code-block:: bash

            python -m monai.apps.nnunet nnUNetV2Runner find_best_configuration --input_config "./input.yaml"

        - predict, ensemble, and post-process

        .. code-block:: bash

            python -m monai.apps.nnunet nnUNetV2Runner predict_ensemble_postprocessing --input_config "./input.yaml"

    """

    def __init__(
        self,
        input_config: Any,
        trainer_class_name: str = "nnUNetTrainer",
        work_dir: str = "work_dir",
        export_validation_probabilities: bool = True,
    ) -> None:
        self.input_info: dict = {}
        self.input_config_or_dict = input_config
        self.trainer_class_name = trainer_class_name
        self.export_validation_probabilities = export_validation_probabilities
        self.work_dir = work_dir

        if isinstance(self.input_config_or_dict, dict):
            self.input_info = self.input_config_or_dict
        elif isinstance(self.input_config_or_dict, str) and os.path.isfile(self.input_config_or_dict):
            self.input_info = ConfigParser.load_config_file(self.input_config_or_dict)
        else:
            raise ValueError(f"{input_config} is not a valid file or dict")

        self.nnunet_raw = self.input_info.pop("nnunet_raw", os.path.join(".", self.work_dir, "nnUNet_raw_data_base"))
        self.nnunet_preprocessed = self.input_info.pop(
            "nnunet_preprocessed", os.path.join(".", self.work_dir, "nnUNet_preprocessed")
        )
        self.nnunet_results = self.input_info.pop(
            "nnunet_results", os.path.join(".", self.work_dir, "nnUNet_trained_models")
        )

        if not os.path.exists(self.nnunet_raw):
            os.makedirs(self.nnunet_raw)

        if not os.path.exists(self.nnunet_preprocessed):
            os.makedirs(self.nnunet_preprocessed)

        if not os.path.exists(self.nnunet_results):
            os.makedirs(self.nnunet_results)

        # claim environment variable
        os.environ["nnUNet_raw"] = self.nnunet_raw
        os.environ["nnUNet_preprocessed"] = self.nnunet_preprocessed
        os.environ["nnUNet_results"] = self.nnunet_results
        os.environ["OMP_NUM_THREADS"] = str(1)

        # dataset_name_or_id has to be a string
        self.dataset_name_or_id = str(self.input_info.pop("dataset_name_or_id", 1))

        try:
            from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name

            self.dataset_name = maybe_convert_to_dataset_name(int(self.dataset_name_or_id))
        except BaseException:
            logger.warning(
                f"Dataset with name/ID: {self.dataset_name_or_id} cannot be found in the record. "
                "Please ignore the message above if you are running the pipeline from a fresh start. "
                "But if the dataset is expected to be found, please check your input_config."
            )

        from nnunetv2.configuration import default_num_processes

        self.default_num_processes = default_num_processes

        self.num_folds = 5
        self.best_configuration: dict = {}

    def convert_dataset(self):
        """Convert and make a copy the dataset to meet the requirements of nnU-Net workflow."""
        try:
            raw_data_foldername_prefix = str(int(self.dataset_name_or_id) + 1000)
            raw_data_foldername_prefix = "Dataset" + raw_data_foldername_prefix[-3:]

            # check if the dataset is created
            subdirs = glob.glob(f"{self.nnunet_raw}/*")
            dataset_ids = [_item.split(os.sep)[-1] for _item in subdirs]
            dataset_ids = [_item.split("_")[0] for _item in dataset_ids]
            if raw_data_foldername_prefix in dataset_ids:
                logger.warning("Dataset with the same ID exists!")
                return

            data_dir = self.input_info.pop("dataroot")
            if data_dir[-1] == os.sep:
                data_dir = data_dir[:-1]

            raw_data_foldername = raw_data_foldername_prefix + "_" + data_dir.split(os.sep)[-1]
            raw_data_foldername = os.path.join(self.nnunet_raw, raw_data_foldername)
            if not os.path.exists(raw_data_foldername):
                os.makedirs(raw_data_foldername)

            from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name

            self.dataset_name = maybe_convert_to_dataset_name(int(self.dataset_name_or_id))

            datalist_json = ConfigParser.load_config_file(self.input_info.pop("datalist"))

            if "training" in datalist_json:
                os.makedirs(os.path.join(raw_data_foldername, "imagesTr"))
                os.makedirs(os.path.join(raw_data_foldername, "labelsTr"))
            else:
                logger.error("The datalist file has incorrect format: the `training` key is not found.")
                return

            test_key = None
            if "test" in datalist_json or "testing" in datalist_json:
                os.makedirs(os.path.join(raw_data_foldername, "imagesTs"))
                test_key = "test" if "test" in datalist_json else "testing"
                if isinstance(datalist_json[test_key][0], dict) and "label" in datalist_json[test_key][0]:
                    os.makedirs(os.path.join(raw_data_foldername, "labelsTs"))

            num_input_channels, num_foreground_classes = analyze_data(datalist_json=datalist_json, data_dir=data_dir)

            modality = self.input_info.pop("modality")
            if not isinstance(modality, list):
                modality = [modality]

            create_new_dataset_json(
                modality=modality,
                num_foreground_classes=num_foreground_classes,
                num_input_channels=num_input_channels,
                num_training_data=len(datalist_json["training"]),
                output_filepath=os.path.join(raw_data_foldername, "dataset.json"),
            )

            create_new_data_copy(
                test_key=test_key,  # type: ignore
                datalist_json=datalist_json,
                data_dir=data_dir,
                num_input_channels=num_input_channels,
                output_datafolder=raw_data_foldername,
            )
        except BaseException as err:
            logger.warning(f"Input config may be incorrect. Detail info: error/exception message is:\n {err}")
            return

    def convert_msd_dataset(self, data_dir: str, overwrite_id: str | None = None, n_proc: int = -1) -> None:
        """
        Convert and make a copy the MSD dataset to meet requirements of nnU-Net workflow.

        Args:
            data_dir: downloaded and extracted MSD dataset folder. CANNOT be nnUNetv1 dataset!
                Example: "/workspace/downloads/Task05_Prostate".
            overwrite_id: Overwrite the dataset id. If not set then use the id of the MSD task (inferred from
                the folder name). Only use this if you already have an equivalently numbered dataset!
            n_proc: Number of processes used.
        """
        from nnunetv2.dataset_conversion.convert_MSD_dataset import convert_msd_dataset

        num_processes = None if n_proc < 0 else self.default_num_processes
        convert_msd_dataset(data_dir, overwrite_id, num_processes)

    def extract_fingerprints(
        self,
        fpe: str = "DatasetFingerprintExtractor",
        npfp: int = -1,
        verify_dataset_integrity: bool = False,
        clean: bool = False,
        verbose: bool = False,
    ) -> None:
        """
        Extracts the dataset fingerprint used for experiment planning.

        Args:
            fpe: [OPTIONAL] Name of the Dataset Fingerprint Extractor class that should be used. Default is
                "DatasetFingerprintExtractor".
            npfp: [OPTIONAL] Number of processes used for fingerprint extraction.
            verify_dataset_integrity: [RECOMMENDED] set this flag to check the dataset integrity. This is
                useful and should be done once for each dataset!
            clean: [OPTIONAL] Set this flag to overwrite existing fingerprints. If this flag is not set and a
                fingerprint already exists, the fingerprint extractor will not run.
            verbose: set this to print a lot of stuff. Useful for debugging. Will disable progress bar!
                Recommended for cluster environments.
        """
        from nnunetv2.experiment_planning.plan_and_preprocess_api import extract_fingerprints

        npfp = self.default_num_processes if npfp < 0 else npfp

        logger.info("Fingerprint extraction...")
        extract_fingerprints([int(self.dataset_name_or_id)], fpe, npfp, verify_dataset_integrity, clean, verbose)

    def plan_experiments(
        self,
        pl: str = "ExperimentPlanner",
        gpu_memory_target: float = 8,
        preprocessor_name: str = "DefaultPreprocessor",
        overwrite_target_spacing: Any = None,
        overwrite_plans_name: str = "nnUNetPlans",
    ) -> None:
        """
        Generate a configuration file that specifies the details of the experiment.

        Args:
            pl: [OPTIONAL] Name of the Experiment Planner class that should be used. Default is "ExperimentPlanner".
                Note: There is no longer a distinction between 2d and 3d planner. It's an all-in-one solution now.
            gpu_memory_target: [OPTIONAL] DANGER ZONE! Sets a custom GPU memory target. Default: 8 [GB].
                Changing this will affect patch and batch size and will definitely affect your models' performance!
                Only use this if you really know what you are doing and NEVER use this without running the
                default nnU-Net first (as a baseline).
            preprocessor_name: [OPTIONAL] DANGER ZONE! Sets a custom preprocessor class. This class must be located in
                nnunetv2.preprocessing. Default: "DefaultPreprocessor". Changing this may affect your models'
                performance! Only use this if you really know what you are doing and NEVER use this without running the
                default nnU-Net first (as a baseline).
            overwrite_target_spacing: [OPTIONAL] DANGER ZONE! Sets a custom target spacing for the 3d_fullres
                and 3d_cascade_fullres configurations. Default: None [no changes]. Changing this will affect
                image size and potentially patch and batch size. This will definitely affect your models' performance!
                Only use this if you really know what you are doing and NEVER use this without running the
                default nnU-Net first (as a baseline). Changing the target spacing for the other configurations
                is currently not implemented. New target spacing must be a list of three numbers!
            overwrite_plans_name: [OPTIONAL] DANGER ZONE! If you used -gpu_memory_target, -preprocessor_name or
                -overwrite_target_spacing it is best practice to use -overwrite_plans_name to generate
                a differently named plans file such that the nnunet default plans are not overwritten.
                You will then need to specify your custom plan.
        """
        from nnunetv2.experiment_planning.plan_and_preprocess_api import plan_experiments

        logger.info("Experiment planning...")
        plan_experiments(
            [int(self.dataset_name_or_id)],
            pl,
            gpu_memory_target,
            preprocessor_name,
            overwrite_target_spacing,
            overwrite_plans_name,
        )

    def preprocess(
        self,
        c: tuple = (M.N_2D, M.N_3D_FULLRES, M.N_3D_LOWRES),
        n_proc: tuple = (8, 8, 8),
        overwrite_plans_name: str = "nnUNetPlans",
        verbose: bool = False,
    ) -> None:
        """
        Apply a set of preprocessing operations to the input data before the training.

        Args:
            overwrite_plans_name: [OPTIONAL] You can use this to specify a custom plans file that you may have
                generated.
            c: [OPTIONAL] Configurations for which the preprocessing should be run. Default: 2d 3f_fullres
                3d_lowres. 3d_cascade_fullres does not need to be specified because it uses the data
                from 3f_fullres. Configurations that do not exist for some datasets will be skipped).
            n_proc: [OPTIONAL] Use this to define how many processes are to be used. If this is just one number then
                this number of processes is used for all configurations specified with -c. If it's a
                list of numbers this list must have as many elements as there are configurations. We
                then iterate over zip(configs, num_processes) to determine the number of processes
                used for each configuration. More processes are always faster (up to the number of
                threads your PC can support, so 8 for a 4-core CPU with hyperthreading. If you don't
                know what that is then don't touch it, or at least don't increase it!). DANGER: More
                often than not the number of processes that can be used is limited by the amount of
                RAM available. Image resampling takes up a lot of RAM. MONITOR RAM USAGE AND
                DECREASE -n_proc IF YOUR RAM FILLS UP TOO MUCH! Default: 8 4 8 (=8 processes for 2d, 4
                for 3d_fullres and 8 for 3d_lowres if -c is at its default).
            verbose: Set this to print a lot of stuff. Useful for debugging. Will disable the progress bar!
                Recommended for cluster environments.
        """
        from nnunetv2.experiment_planning.plan_and_preprocess_api import preprocess

        logger.info("Preprocessing...")
        preprocess(
            [int(self.dataset_name_or_id)],
            overwrite_plans_name,
            configurations=c,
            num_processes=n_proc,
            verbose=verbose,
        )

    def plan_and_process(
        self,
        fpe: str = "DatasetFingerprintExtractor",
        npfp: int = 8,
        verify_dataset_integrity: bool = False,
        no_pp: bool = False,
        clean: bool = False,
        pl: str = "ExperimentPlanner",
        gpu_memory_target: int = 8,
        preprocessor_name: str = "DefaultPreprocessor",
        overwrite_target_spacing: Any = None,
        overwrite_plans_name: str = "nnUNetPlans",
        c: tuple = (M.N_2D, M.N_3D_FULLRES, M.N_3D_LOWRES),
        n_proc: tuple = (8, 8, 8),
        verbose: bool = False,
    ) -> None:
        """
        Performs experiment planning and preprocessing before the training.

        Args:
            fpe: [OPTIONAL] Name of the Dataset Fingerprint Extractor class that should be used. Default is
                "DatasetFingerprintExtractor".
            npfp: [OPTIONAL] Number of processes used for fingerprint extraction. Default: 8.
            verify_dataset_integrity: [RECOMMENDED] set this flag to check the dataset integrity.
                This is useful and should be done once for each dataset!
            no_pp: [OPTIONAL] Set this to only run fingerprint extraction and experiment planning (no
                preprocessing). Useful for debugging.
            clean:[OPTIONAL] Set this flag to overwrite existing fingerprints. If this flag is not set and a
                fingerprint already exists, the fingerprint extractor will not run. REQUIRED IF YOU
                CHANGE THE DATASET FINGERPRINT EXTRACTOR OR MAKE CHANGES TO THE DATASET!
            pl: [OPTIONAL] Name of the Experiment Planner class that should be used. Default is "ExperimentPlanner".
                Note: There is no longer a distinction between 2d and 3d planner. It's an all-in-one solution now.
            gpu_memory_target: [OPTIONAL] DANGER ZONE! Sets a custom GPU memory target. Default: 8 [GB].
                Changing this will affect patch and batch size and will
                definitely affect your models' performance! Only use this if you really know what you
                are doing and NEVER use this without running the default nnU-Net first (as a baseline).
            preprocessor_name: [OPTIONAL] DANGER ZONE! Sets a custom preprocessor class. This class must be located in
                nnunetv2.preprocessing. Default: "DefaultPreprocessor". Changing this may affect your
                models' performance! Only use this if you really know what you
                are doing and NEVER use this without running the default nnU-Net first (as a baseline).
            overwrite_target_spacing: [OPTIONAL] DANGER ZONE! Sets a custom target spacing for the 3d_fullres and
                3d_cascade_fullres configurations. Default: None [no changes]. Changing this will affect image size and
                potentially patch and batch size. This will definitely affect your models performance!
                Only use this if you really know what you are doing and NEVER use this without running the
                default nnU-Net first (as a baseline). Changing the target spacing for the other
                configurations is currently not implemented. New target spacing must be a list of three numbers!
            overwrite_plans_name: [OPTIONAL] USE A CUSTOM PLANS IDENTIFIER. If you used -gpu_memory_target,
                -preprocessor_name or -overwrite_target_spacing it is best practice to use -overwrite_plans_name to
                generate a differently named plans file such that the nnunet default plans are not
                overwritten. You will then need to specify your custom plans file with -p whenever
                running other nnunet commands (training, inference, etc)
            c: [OPTIONAL] Configurations for which the preprocessing should be run. Default: 2d 3f_fullres
                3d_lowres. 3d_cascade_fullres does not need to be specified because it uses the data
                from 3f_fullres. Configurations that do not exist for some datasets will be skipped.
            n_proc: [OPTIONAL] Use this to define how many processes are to be used. If this is just one number then
                this number of processes is used for all configurations specified with -c. If it's a
                list of numbers this list must have as many elements as there are configurations. We
                then iterate over zip(configs, num_processes) to determine the number of processes
                used for each configuration. More processes are always faster (up to the number of
                threads your PC can support, so 8 for a 4-core CPU with hyperthreading. If you don't
                know what that is then don't touch it, or at least don't increase it!). DANGER: More
                often than not the number of processes that can be used is limited by the amount of
                RAM available. Image resampling takes up a lot of RAM. MONITOR RAM USAGE AND
                DECREASE -n_proc IF YOUR RAM FILLS UP TOO MUCH! Default: 8 4 8 (=8 processes for 2d, 4
                for 3d_fullres and 8 for 3d_lowres if -c is at its default).
            verbose: Set this to print a lot of stuff. Useful for debugging. Will disable progress bar!
                (Recommended for cluster environments).
        """
        self.extract_fingerprints(fpe, npfp, verify_dataset_integrity, clean, verbose)
        self.plan_experiments(pl, gpu_memory_target, preprocessor_name, overwrite_target_spacing, overwrite_plans_name)

        if not no_pp:
            self.preprocess(c, n_proc, overwrite_plans_name, verbose)

    def train_single_model(self, config: Any, fold: int, gpu_id: tuple | list | int = 0, **kwargs: Any) -> None:
        """
        Run the training on a single GPU with one specified configuration provided.
        Note: this will override the environment variable `CUDA_VISIBLE_DEVICES`.

        Args:
            config: configuration that should be trained. Examples: "2d", "3d_fullres", "3d_lowres".
            fold: fold of the 5-fold cross-validation. Should be an int between 0 and 4.
            gpu_id: an integer to select the device to use, or a tuple/list of GPU device indices used for multi-GPU
                training (e.g., (0,1)). Default: 0.
        from nnunetv2.run.run_training import run_training
            kwargs: this optional parameter allows you to specify additional arguments in
                ``nnunetv2.run.run_training.run_training``. Currently supported args are
                    - plans_identifier: custom plans identifier. Default: "nnUNetPlans".
                    - pretrained_weights: path to nnU-Net checkpoint file to be used as pretrained model. Will only be
                        used when actually training. Beta. Use with caution. Default: False.
                    - use_compressed_data: True to use compressed data for training. Reading compressed data is much
                        more CPU and (potentially) RAM intensive and should only be used if you know what you are
                        doing. Default: False.
                    - continue_training: continue training from latest checkpoint. Default: False.
                    - only_run_validation: True to run the validation only. Requires training to have finished.
                        Default: False.
                    - disable_checkpointing: True to disable checkpointing. Ideal for testing things out and you
                        don't want to flood your hard drive with checkpoints. Default: False.
        """
        if "num_gpus" in kwargs:
            kwargs.pop("num_gpus")
            logger.warning("please use gpu_id to set the GPUs to use")

        if "trainer_class_name" in kwargs:
            kwargs.pop("trainer_class_name")
            logger.warning("please specify the `trainer_class_name` in the __init__ of `nnUNetV2Runner`.")

        if "export_validation_probabilities" in kwargs:
            kwargs.pop("export_validation_probabilities")
            logger.warning("please specify the `export_validation_probabilities` in the __init__ of `nnUNetV2Runner`.")

        if isinstance(gpu_id, (tuple, list)):
            if len(gpu_id) > 1:
                gpu_ids_str = ""
                for _i in range(len(gpu_id)):
                    gpu_ids_str += f"{gpu_id[_i]},"
                os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids_str[:-1]
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_id[0]}"
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_id}"

        from nnunetv2.run.run_training import run_training

        if isinstance(gpu_id, int) or len(gpu_id) == 1:
            run_training(
                dataset_name_or_id=self.dataset_name_or_id,
                configuration=config,
                fold=fold,
                trainer_class_name=self.trainer_class_name,
                export_validation_probabilities=self.export_validation_probabilities,
                **kwargs,
            )
        else:
            run_training(
                dataset_name_or_id=self.dataset_name_or_id,
                configuration=config,
                fold=fold,
                num_gpus=len(gpu_id),
                trainer_class_name=self.trainer_class_name,
                export_validation_probabilities=self.export_validation_probabilities,
                **kwargs,
            )

    def train(
        self,
        configs: tuple | str = (M.N_3D_FULLRES, M.N_2D, M.N_3D_LOWRES, M.N_3D_CASCADE_FULLRES),
        gpu_id_for_all: tuple | list | int | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Run the training for all the models specified by the configurations.
        Note: to set the number of GPUs to use, use ``gpu_id_for_all`` instead of the `CUDA_VISIBLE_DEVICES`
        environment variable.

        Args:
            configs: configurations that should be trained.
                Default: ("2d", "3d_fullres", "3d_lowres", "3d_cascade_fullres").
            gpu_id_for_all: a tuple/list/integer of GPU device ID(s) to use for the training. Default:
                None (all available GPUs).
            kwargs: this optional parameter allows you to specify additional arguments defined in the
                ``train_single_model`` method.
        """
        if gpu_id_for_all is None:
            result = subprocess.run(["nvidia-smi", "--list-gpus"], stdout=subprocess.PIPE)
            output = result.stdout.decode("utf-8")
            num_gpus = len(output.strip().split("\n"))
            gpu_id_for_all = tuple(range(num_gpus))
        elif isinstance(gpu_id_for_all, int):
            gpu_id_for_all = ensure_tuple(gpu_id_for_all)
        logger.info(f"number of GPUs is {len(gpu_id_for_all)}, device ids are {gpu_id_for_all}")
        if len(gpu_id_for_all) > 1:
            self.train_parallel(configs=ensure_tuple(configs), gpu_id_for_all=gpu_id_for_all, **kwargs)
        else:
            for cfg in ensure_tuple(configs):
                for _fold in range(self.num_folds):
                    self.train_single_model(config=cfg, fold=_fold, gpu_id=gpu_id_for_all, **kwargs)

    def train_parallel_cmd(
        self,
        configs: tuple | str = (M.N_3D_FULLRES, M.N_2D, M.N_3D_LOWRES, M.N_3D_CASCADE_FULLRES),
        gpu_id_for_all: tuple | list | int | None = None,
        **kwargs: Any,
    ) -> list:
        """
        Create the line command for subprocess call for parallel training.

        Args:
            configs: configurations that should be trained.
                Default: ("2d", "3d_fullres", "3d_lowres", "3d_cascade_fullres").
            gpu_id_for_all: a tuple/list/integer of GPU device ID(s) to use for the training. Default:
                None (all available GPUs).
            kwargs: this optional parameter allows you to specify additional arguments defined in the
                ``train_single_model`` method.
        """
        # unpack compressed files
        folder_names = []
        for root, _, files in os.walk(os.path.join(self.nnunet_preprocessed, self.dataset_name)):
            if any(file.endswith(".npz") for file in files):
                folder_names.append(root)

        from nnunetv2.training.dataloading.utils import unpack_dataset

        for folder_name in folder_names:
            logger.info(f"unpacking '{folder_name}'...")
            unpack_dataset(
                folder=folder_name,
                unpack_segmentation=True,
                overwrite_existing=False,
                num_processes=self.default_num_processes,
            )

        # model training
        kwargs = kwargs or {}
        devices = ensure_tuple(gpu_id_for_all)
        n_devices = len(devices)
        _configs = [[M.N_3D_FULLRES, M.N_2D, M.N_3D_LOWRES], [M.N_3D_CASCADE_FULLRES]]
        all_cmds: list = []
        for _stage in range(len(_configs)):
            all_cmds.append({_j: [] for _j in devices})
            _index = 0

            for _config in _configs[_stage]:
                if _config in ensure_tuple(configs):
                    for _i in range(self.num_folds):
                        the_device = gpu_id_for_all[_index % n_devices]  # type: ignore
                        cmd = (
                            "python -m monai.apps.nnunet nnUNetV2Runner train_single_model "
                            + f"--input_config '{self.input_config_or_dict}' --work_dir '{self.work_dir}' "
                            + f"--config '{_config}' --fold {_i} --gpu_id {the_device} "
                            + f"--trainer_class_name {self.trainer_class_name} "
                            + f"--export_validation_probabilities {self.export_validation_probabilities}"
                        )
                        for _key, _value in kwargs.items():
                            cmd += f" --{_key} {_value}"
                        all_cmds[-1][the_device].append(cmd)
                        _index += 1
        return all_cmds

    def train_parallel(
        self,
        configs: tuple | str = (M.N_3D_FULLRES, M.N_2D, M.N_3D_LOWRES, M.N_3D_CASCADE_FULLRES),
        gpu_id_for_all: tuple | list | int | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Create the line command for subprocess call for parallel training.
        Note: to set the number of GPUs to use, use ``gpu_id_for_all`` instead of the `CUDA_VISIBLE_DEVICES`
        environment variable.

        Args:
            configs: configurations that should be trained.
                default: ("2d", "3d_fullres", "3d_lowres", "3d_cascade_fullres").
            gpu_id_for_all: a tuple/list/integer of GPU device ID(s) to use for the training. Default:
                None (all available GPUs).
            kwargs: this optional parameter allows you to specify additional arguments defined in the
                ``train_single_model`` method.
        """
        all_cmds = self.train_parallel_cmd(configs=configs, gpu_id_for_all=gpu_id_for_all, **kwargs)
        for s, cmds in enumerate(all_cmds):
            for gpu_id, gpu_cmd in cmds.items():
                if not gpu_cmd:
                    continue
                logger.info(
                    f"training - stage {s + 1}:\n"
                    f"for gpu {gpu_id}, commands: {gpu_cmd}\n"
                    f"log '.txt' inside '{os.path.join(self.nnunet_results, self.dataset_name)}'"
                )
        for stage in all_cmds:
            processes = []
            for device_id in stage:
                if not stage[device_id]:
                    continue
                cmd_str = "; ".join(stage[device_id])
                logger.info(f"Current running command on GPU device {device_id}:\n{cmd_str}\n")
                processes.append(subprocess.Popen(cmd_str, shell=True, stdout=subprocess.DEVNULL))
            # finish this stage first
            for p in processes:
                p.wait()

    def validate_single_model(self, config: str, fold: int, **kwargs: Any) -> None:
        """
        Perform validation on single model.

        Args:
            config: configuration that should be trained.
            fold: fold of the 5-fold cross-validation. Should be an int between 0 and 4.
            kwargs: this optional parameter allows you to specify additional arguments defined in the
                ``train_single_model`` method.
        """
        self.train_single_model(config=config, fold=fold, only_run_validation=True, **kwargs)

    def validate(
        self, configs: tuple = (M.N_3D_FULLRES, M.N_2D, M.N_3D_LOWRES, M.N_3D_CASCADE_FULLRES), **kwargs: Any
    ) -> None:
        """
        Perform validation in all models defined by the configurations over 5 folds.

        Args:
            configs: configurations that should be trained.
                default: ("2d", "3d_fullres", "3d_lowres", "3d_cascade_fullres").
            kwargs: this optional parameter allows you to specify additional arguments defined in the
                ``train_single_model`` method.
        """
        for cfg in ensure_tuple(configs):
            for _fold in range(self.num_folds):
                self.validate_single_model(config=cfg, fold=_fold, **kwargs)

    def find_best_configuration(
        self,
        plans: tuple | str = "nnUNetPlans",
        configs: tuple | str = (M.N_2D, M.N_3D_FULLRES, M.N_3D_LOWRES, M.N_3D_CASCADE_FULLRES),
        trainers: tuple | str | None = None,
        allow_ensembling: bool = True,
        num_processes: int = -1,
        overwrite: bool = True,
        folds: list[int] | tuple[int, ...] = (0, 1, 2, 3, 4),
        strict: bool = False,
    ) -> None:
        """
        Find the best model configurations.

        Args:
            plans: list of plan identifiers. Default: nnUNetPlans.
            configs: list of configurations. Default: ["2d", "3d_fullres", "3d_lowres", "3d_cascade_fullres"].
            trainers: list of trainers. Default: nnUNetTrainer.
            allow_ensembling: set this flag to enable ensembling.
            num_processes: number of processes to use for ensembling, postprocessing, etc.
            overwrite: if set we will overwrite already ensembled files etc. May speed up consecutive
                runs of this command (not recommended) at the risk of not updating outdated results.
            folds: folds to use. Default: (0, 1, 2, 3, 4).
            strict: a switch that triggers RunTimeError if the logging folder cannot be found. Default: False.
        """
        from nnunetv2.evaluation.find_best_configuration import (
            dumb_trainer_config_plans_to_trained_models_dict,
            find_best_configuration,
        )

        configs = ensure_tuple(configs)
        plans = ensure_tuple(plans)

        if trainers is None:
            trainers = self.trainer_class_name
        trainers = ensure_tuple(trainers)

        models = dumb_trainer_config_plans_to_trained_models_dict(trainers, configs, plans)
        num_processes = self.default_num_processes if num_processes < 0 else num_processes
        find_best_configuration(
            int(self.dataset_name_or_id),
            models,
            allow_ensembling=allow_ensembling,
            num_processes=num_processes,
            overwrite=overwrite,
            folds=folds,
            strict=strict,
        )

    def predict(
        self,
        list_of_lists_or_source_folder: str | list[list[str]],
        output_folder: str | None | list[str],
        model_training_output_dir: str,
        use_folds: tuple[int, ...] | str | None = None,
        tile_step_size: float = 0.5,
        use_gaussian: bool = True,
        use_mirroring: bool = True,
        perform_everything_on_gpu: bool = True,
        verbose: bool = True,
        save_probabilities: bool = False,
        overwrite: bool = True,
        checkpoint_name: str = "checkpoint_final.pth",
        folder_with_segs_from_prev_stage: str | None = None,
        num_parts: int = 1,
        part_id: int = 0,
        num_processes_preprocessing: int = -1,
        num_processes_segmentation_export: int = -1,
        gpu_id: int = 0,
    ) -> None:
        """
        Use this to run inference with nnU-Net. This function is used when you want to manually specify a folder containing
            a trained nnU-Net model. This is useful when the nnunet environment variables (nnUNet_results) are not set.

        Args:
            list_of_lists_or_source_folder: input folder. Remember to use the correct channel numberings for
                your files (_0000 etc). File endings must be the same as the training dataset!
            output_folder: Output folder. If it does not exist it will be created. Predicted segmentations will
                have the same name as their source images.
            model_training_output_dir: folder in which the trained model is. Must have subfolders fold_X for the
                different folds you trained.
            use_folds: specify the folds of the trained model that should be used for prediction
                Default: (0, 1, 2, 3, 4).
            tile_step_size: step size for sliding window prediction. The larger it is the faster but less accurate
                the prediction. Default: 0.5. Cannot be larger than 1. We recommend the default.
            use_gaussian: use Gaussian smoothing as test-time augmentation.
            use_mirroring: use mirroring/flipping as test-time augmentation.
            verbose: set this if you like being talked to. You will have to be a good listener/reader.
            save_probabilities: set this to export predicted class "probabilities". Required if you want to ensemble
                multiple configurations.
            overwrite: overwrite an existing previous prediction (will not overwrite existing files)
            checkpoint_name: name of the checkpoint you want to use. Default: checkpoint_final.pth.
            folder_with_segs_from_prev_stage: folder containing the predictions of the previous stage.
                Required for cascaded models.
            num_parts: number of separate nnUNetv2_predict call that you will be making. Default: 1 (= this one
                call predicts everything).
            part_id: if multiple nnUNetv2_predict exist, which one is this? IDs start with 0 can end with
                num_parts - 1. So when you submit 5 nnUNetv2_predict calls you need to set -num_parts
                5 and use -part_id 0, 1, 2, 3 and 4.
            num_processes_preprocessing: out-of-RAM issues.
            num_processes_segmentation_export: Number of processes used for segmentation export.
                More is not always better. Beware of out-of-RAM issues.
            gpu_id: which GPU to use for prediction.
        """
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_id}"

        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

        n_processes_preprocessing = (
            self.default_num_processes if num_processes_preprocessing < 0 else num_processes_preprocessing
        )
        n_processes_segmentation_export = (
            self.default_num_processes if num_processes_segmentation_export < 0 else num_processes_segmentation_export
        )
        predictor = nnUNetPredictor(
            tile_step_size=tile_step_size,
            use_gaussian=use_gaussian,
            use_mirroring=use_mirroring,
            perform_everything_on_device=perform_everything_on_gpu,
            verbose=verbose,
        )
        predictor.initialize_from_trained_model_folder(
            model_training_output_dir=model_training_output_dir, use_folds=use_folds, checkpoint_name=checkpoint_name
        )
        predictor.predict_from_files(
            list_of_lists_or_source_folder=list_of_lists_or_source_folder,
            output_folder_or_list_of_truncated_output_files=output_folder,
            save_probabilities=save_probabilities,
            overwrite=overwrite,
            num_processes_preprocessing=n_processes_preprocessing,
            num_processes_segmentation_export=n_processes_segmentation_export,
            folder_with_segs_from_prev_stage=folder_with_segs_from_prev_stage,
            num_parts=num_parts,
            part_id=part_id,
        )

    def predict_ensemble_postprocessing(
        self,
        folds: tuple = (0, 1, 2, 3, 4),
        run_ensemble: bool = True,
        run_predict: bool = True,
        run_postprocessing: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Run prediction, ensemble, and/or postprocessing optionally.

        Args:
            folds: which folds to use
            run_ensemble: whether to run ensembling.
            run_predict: whether to predict using trained checkpoints
            run_postprocessing: whether to conduct post-processing
            kwargs: this optional parameter allows you to specify additional arguments defined in the
                ``predict`` method.
        """
        from nnunetv2.ensembling.ensemble import ensemble_folders
        from nnunetv2.postprocessing.remove_connected_components import apply_postprocessing_to_folder
        from nnunetv2.utilities.file_path_utilities import get_output_folder

        source_dir = join(self.nnunet_raw, self.dataset_name, "imagesTs")
        target_dir_base = join(self.nnunet_results, self.dataset_name)

        self.best_configuration = ConfigParser.load_config_file(
            os.path.join(self.nnunet_results, self.dataset_name, "inference_information.json")
        )

        run_ensemble = (
            run_ensemble and len(self.best_configuration["best_model_or_ensemble"]["selected_model_or_models"]) > 1
        )

        used_folds = folds
        output_folders = []
        for im in self.best_configuration["best_model_or_ensemble"]["selected_model_or_models"]:
            output_dir = join(target_dir_base, f"pred_{im['configuration']}")
            output_folders.append(output_dir)

            if run_predict:
                model_folder = get_output_folder(
                    int(self.dataset_name_or_id), im["trainer"], im["plans_identifier"], im["configuration"]
                )
                self.predict(
                    list_of_lists_or_source_folder=source_dir,
                    output_folder=output_dir,
                    model_training_output_dir=model_folder,
                    use_folds=used_folds,
                    save_probabilities=run_ensemble,
                    verbose=False,
                    overwrite=True,
                    **kwargs,
                )

        # if we have an ensemble, we need to ensemble the results
        if run_ensemble:
            ensemble_folders(
                output_folders, join(target_dir_base, "ensemble_predictions"), save_merged_probabilities=False
            )
            if run_postprocessing:
                folder_for_pp = join(target_dir_base, "ensemble_predictions")
        elif run_postprocessing:
            folder_for_pp = output_folders[0]

        # apply postprocessing
        if run_postprocessing:
            pp_fns, pp_fn_kwargs = load_pickle(self.best_configuration["best_model_or_ensemble"]["postprocessing_file"])
            apply_postprocessing_to_folder(
                folder_for_pp,
                join(target_dir_base, "ensemble_predictions_postprocessed"),
                pp_fns,
                pp_fn_kwargs,
                plans_file_or_dict=self.best_configuration["best_model_or_ensemble"]["some_plans_file"],
            )

    def run(
        self,
        run_convert_dataset: bool = True,
        run_plan_and_process: bool = True,
        run_train: bool = True,
        run_find_best_configuration: bool = True,
        run_predict_ensemble_postprocessing: bool = True,
    ) -> None:
        """
        Run the nnU-Net pipeline.

        Args:
            run_convert_dataset: whether to convert datasets, defaults to True.
            run_plan_and_process: whether to preprocess and analyze the dataset, defaults to True.
            run_train: whether to train models, defaults to True.
            run_find_best_configuration: whether to find the best model (ensemble) configurations, defaults to True.
            run_predict_ensemble_postprocessing: whether to make predictions on test datasets, defaults to True.
        """
        if run_convert_dataset:
            self.convert_dataset()

        if run_plan_and_process:
            self.plan_and_process()

        if run_train:
            self.train()

        if run_find_best_configuration:
            self.find_best_configuration()

        if run_predict_ensemble_postprocessing:
            self.predict_ensemble_postprocessing()

        return

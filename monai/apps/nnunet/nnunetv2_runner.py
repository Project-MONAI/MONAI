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

import copy
import glob
import os
import subprocess

import numpy as np

import monai
from monai.bundle import ConfigParser
from monai.utils import optional_import

load_pickle, _ = optional_import("batchgenerators.utilities.file_and_folder_operations", name="load_pickle")
join, _ = optional_import("batchgenerators.utilities.file_and_folder_operations", name="join")
tqdm, has_tqdm = optional_import("tqdm", name="tqdm")
nib, _ = optional_import("nibabel", name="nib")


class nnUNetV2Runner:
    def __init__(self, input, work_dir: str = "work_dir"):
        """
        An interface for handling nnU-Net V2 with minimal inputs and understanding of the internal states in nnU-Net.
        The users can run the nnU-Net V2 with default settings in one line of code. They can also run parts of the process
        of nnU-Net V2 instead of the complete pipeline.

        The output of the interface is a directory that contains
            - converted dataset met requirement of nnU-Net V2
            - data analysis results
            - checkpoints from the trained U-Net models
            - validation accuracy in each fold of cross-validation
            - the predictions on the testing datasets from the final algorithm ensemble and potential post-processing

        Args:
            work_dir: working directory to save the intermediate and final results.
            input: the configuration dictionary or the file path to the configuration in form of YAML.
                The configuration should contain datalist, dataroot, modality.
        """
        self.input_info = []
        self.input_config_or_dict = input
        self.work_dir = work_dir

        if isinstance(self.input_config_or_dict, dict):
            self.input_info = self.input_config_or_dict
        elif isinstance(self.input_config_or_dict, str) and os.path.isfile(self.input_config_or_dict):
            self.input_info = ConfigParser.load_config_file(self.input_config_or_dict)
        else:
            raise ValueError(f"{input} is not a valid file or dict")

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
            print("[warning] Dataset ID does not exist! Check input '.yaml' if this is unexpected.")

        from nnunetv2.configuration import default_num_processes

        self.default_num_processes = default_num_processes

        self.num_folds = 5
        self.best_configuration = None

        result = subprocess.run(["nvidia-smi", "--list-gpus"], stdout=subprocess.PIPE)
        output = result.stdout.decode("utf-8")
        self.num_gpus = len(output.strip().split("\n"))
        print(f"[info] number of gpus is {self.num_gpus}")

    def convert_dataset(self):
        try:
            raw_data_foldername_perfix = str(int(self.dataset_name_or_id) + 1000)
            raw_data_foldername_perfix = "Dataset" + raw_data_foldername_perfix[-3:]

            # check if dataset is created
            subdirs = glob.glob(f"{self.nnunet_raw}/*")
            dataset_ids = [_item.split(os.sep)[-1] for _item in subdirs]
            dataset_ids = [_item.split("_")[0] for _item in dataset_ids]
            if raw_data_foldername_perfix in dataset_ids:
                print("Dataset with the same ID exists!")
                return

            data_dir = self.input_info.pop("dataroot")
            if data_dir[-1] == os.sep:
                data_dir = data_dir[:-1]

            raw_data_foldername = raw_data_foldername_perfix + "_" + data_dir.split(os.sep)[-1]
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
                print("Input '.json' data list is incorrect.")
                return

            test_key = None
            if "test" in datalist_json or "testing" in datalist_json:
                os.makedirs(os.path.join(raw_data_foldername, "imagesTs"))
                test_key = "test" if "test" in datalist_json else "testing"
                if isinstance(datalist_json[test_key][0], dict) and "label" in datalist_json[test_key][0]:
                    os.makedirs(os.path.join(raw_data_foldername, "labelsTs"))

            img = monai.transforms.LoadImage(image_only=True, ensure_channel_first=True, simple_keys=True)(
                os.path.join(data_dir, datalist_json["training"][0]["image"])
            )
            num_input_channels = img.size()[0] if img.dim() == 4 else 1
            print(f"[info] num_input_channels: {num_input_channels}")

            num_foreground_classes = 0
            for _i in range(len(datalist_json["training"])):
                seg = monai.transforms.LoadImage(image_only=True, ensure_channel_first=True, simple_keys=True)(
                    os.path.join(data_dir, datalist_json["training"][_i]["label"])
                )
                num_foreground_classes = max(num_foreground_classes, int(seg.max()))
            print(f"[info] num_foreground_classes: {num_foreground_classes}")

            new_json_data = {}

            modality = self.input_info.pop("modality")
            if not isinstance(modality, list):
                modality = [modality]

            new_json_data["channel_names"] = {}
            for _j in range(num_input_channels):
                new_json_data["channel_names"][str(_j)] = modality[_j]

            new_json_data["labels"] = {}
            new_json_data["labels"]["background"] = 0
            for _j in range(num_foreground_classes):
                new_json_data["labels"][f"class{_j + 1}"] = _j + 1

            new_json_data["numTraining"] = len(datalist_json["training"])
            new_json_data["file_ending"] = ".nii.gz"

            ConfigParser.export_config_file(
                config=new_json_data,
                filepath=os.path.join(raw_data_foldername, "dataset.json"),
                fmt="json",
                sort_keys=True,
                indent=4,
                ensure_ascii=False,
            )

            _index = 0
            new_datalist_json = {}
            new_datalist_json["training"] = []
            new_datalist_json[test_key] = []

            for _key, _folder, _label_folder in list(
                zip(["training", test_key], ["imagesTr", "imagesTs"], ["labelsTr", "labelsTs"])
            ):
                if _key is None:
                    continue

                print(f"[info] converting data section: {_key}...")
                for _k in tqdm(range(len(datalist_json[_key]))) if has_tqdm else range(len(datalist_json[_key])):
                    orig_img_name = (
                        datalist_json[_key][_k]["image"]
                        if isinstance(datalist_json[_key][_k], dict)
                        else datalist_json[_key][_k]
                    )
                    img_name = f"case_{_index}"
                    _index += 1

                    # copy image
                    nda = monai.transforms.LoadImage(image_only=True, ensure_channel_first=True, simple_keys=True)(
                        os.path.join(data_dir, orig_img_name)
                    )
                    affine = nda.meta["original_affine"]
                    nda = nda.numpy()
                    for _l in range(num_input_channels):
                        outimg = nib.Nifti1Image(nda[_l, ...], affine)
                        index = "_" + str(_l + 10000)[-4:]
                        nib.save(outimg, os.path.join(raw_data_foldername, _folder, img_name + index + ".nii.gz"))

                    # copy label
                    if isinstance(datalist_json[_key][_k], dict) and "label" in datalist_json[_key][_k]:
                        nda = monai.transforms.LoadImage(image_only=True, ensure_channel_first=True, simple_keys=True)(
                            os.path.join(data_dir, datalist_json[_key][_k]["label"])
                        )
                        affine = nda.meta["original_affine"]
                        nda = nda.numpy().astype(np.uint8)
                        nda = nda[0, ...] if nda.ndim == 4 and nda.shape[0] == 1 else nda
                        nib.save(
                            nib.Nifti1Image(nda, affine),
                            os.path.join(raw_data_foldername, _label_folder, img_name + ".nii.gz"),
                        )

                    if isinstance(datalist_json[_key][_k], dict):
                        _val = copy.deepcopy(datalist_json[_key][_k])
                        _val["new_name"] = img_name
                        new_datalist_json[_key].append(_val)
                    else:
                        new_datalist_json[_key].append({"image": datalist_json[_key][_k], "new_name": img_name})

            ConfigParser.export_config_file(
                config=new_datalist_json,
                filepath=os.path.join(raw_data_foldername, "datalist.json"),
                fmt="json",
                sort_keys=True,
                indent=4,
                ensure_ascii=False,
            )
        except BaseException:
            print("Input '.yaml' is incorrect.")
            return

    def convert_msd_dataset(self, data_dir, overwrite_id=None, np=-1):
        from nnunetv2.dataset_conversion.convert_MSD_dataset import convert_msd_dataset

        num_processes = None if np < 0 else self.default_num_processes
        convert_msd_dataset(data_dir, overwrite_id, num_processes)

    def extract_fingerprints(
        self, fpe="DatasetFingerprintExtractor", npfp=8, verify_dataset_integrity=False, clean=False, verbose=False
    ):
        from nnunetv2.experiment_planning.plan_and_preprocess_api import extract_fingerprints

        print("Fingerprint extraction...")
        extract_fingerprints([int(self.dataset_name_or_id)], fpe, npfp, verify_dataset_integrity, clean, verbose)

    def plan_experiments(
        self,
        pl="ExperimentPlanner",
        gpu_memory_target=8,
        preprocessor_name="DefaultPreprocessor",
        overwrite_target_spacing=None,
        overwrite_plans_name="nnUNetPlans",
        verbose=False,
    ):
        from nnunetv2.experiment_planning.plan_and_preprocess_api import plan_experiments

        print("Experiment planning...")
        plan_experiments(
            [int(self.dataset_name_or_id)],
            pl,
            gpu_memory_target,
            preprocessor_name,
            overwrite_target_spacing,
            overwrite_plans_name,
        )

    def preprocess(
        self, c=["2d", "3d_fullres", "3d_lowres"], np=[8, 8, 8], overwrite_plans_name="nnUNetPlans", verbose=False
    ):
        from nnunetv2.experiment_planning.plan_and_preprocess_api import preprocess

        print("Preprocessing...")
        preprocess(
            [int(self.dataset_name_or_id)], overwrite_plans_name, configurations=c, num_processes=np, verbose=verbose
        )

    def plan_and_process(
        self,
        fpe="DatasetFingerprintExtractor",
        npfp=8,
        verify_dataset_integrity=False,
        no_pp=False,
        clean=False,
        pl="ExperimentPlanner",
        gpu_memory_target=8,
        preprocessor_name="DefaultPreprocessor",
        overwrite_target_spacing=None,
        overwrite_plans_name="nnUNetPlans",
        c=["2d", "3d_fullres", "3d_lowres"],
        np=[8, 8, 8],
        verbose=False,
    ):
        self.extract_fingerprints(fpe, npfp, verify_dataset_integrity, clean, verbose)
        self.plan_experiments(pl, gpu_memory_target, preprocessor_name, overwrite_target_spacing, overwrite_plans_name)
        self.preprocess(c, np, overwrite_plans_name, verbose)

    def train_single_model(self, config, fold, gpu_id: int = 0, **kwargs):
        """
        Args:
            config: configuration that should be trained.
            fold: fold of the 5-fold cross-validation. Should be an int between 0 and 4.
            trainer_class_name: name of the custom trainer class. default: 'nnUNetTrainer'.
            plans_identifier: custom plans identifier. default: 'nnUNetPlans'.
            pretrained_weights: path to nnU-Net checkpoint file to be used as pretrained model. Will only be used
                when actually training. Beta. Use with caution. default: False.
            num_gpus: number of GPUs to use for training. default: 1.
            use_compressed_data: true to use compressed data for training. Reading compressed data is much more CPU and
                (potentially) RAM intensive and should only be used if you know what you are doing default: False.
            export_validation_probabilities: true to save softmax predictions from final validation as npz files
                (in addition to predicted segmentations). Needed for finding the best ensemble. default: False.
            continue_training: continue training from latest checkpoint. default: False.
            only_run_validation: true to run the validation only. Requires training to have finished. default: False.
            disable_checkpointing: true to disable checkpointing. Ideal for testing things out and you dont want to
                flood your hard drive with checkpoints. default: False.
        """
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_id}"

        from nnunetv2.run.run_training import run_training

        run_training(dataset_name_or_id=self.dataset_name_or_id, configuration=config, fold=fold, **kwargs)

    def train(self, configs=["3d_fullres", "2d", "3d_lowres", "3d_cascade_fullres"], **kwargs):
        """
        Args:
            configs: configurations that should be trained.
        """
        if isinstance(configs, str):
            configs = [configs]

        if self.num_gpus > 1:
            self.train_parallel(configs=configs, **kwargs)
        else:
            for _i in range(len(configs)):
                _config = configs[_i]
                for _fold in range(self.num_folds):
                    self.train_single_model(config=_config, fold=_fold, **kwargs)

    def train_parallel(self, configs=["3d_fullres", "2d", "3d_lowres", "3d_cascade_fullres"], **kwargs):
        """
        Args:
            configs: configurations that should be trained.
        """
        if isinstance(configs, str):
            configs = [configs]

        # unpack compressed files
        folder_names = []
        for root, dirs, files in os.walk(os.path.join(self.nnunet_preprocessed, self.dataset_name)):
            if any(file.endswith(".npz") for file in files):
                folder_names.append(root)

        from nnunetv2.training.dataloading.utils import unpack_dataset

        for folder_name in folder_names:
            print(f"[info] unpacking '{folder_name}'...")
            _ = unpack_dataset(
                folder=folder_name,
                unpack_segmentation=True,
                overwrite_existing=False,
                num_processes=self.default_num_processes,
            )

        # model training
        _configs = [["3d_fullres", "2d", "3d_lowres"], ["3d_cascade_fullres"]]
        for _stage in range(2):
            cmds = []
            _index = 0

            for _config in _configs[_stage]:
                if _config in configs:
                    for _i in range(self.num_folds):
                        cmd = (
                            "python -m monai.apps.nnunet nnUNetV2Runner train_single_model "
                            + f"--input '{self.input_config_or_dict}' --config '{_config}' --fold {_i} --gpu_id {_index%self.num_gpus}"
                        )

                        if isinstance(kwargs, dict):
                            for _key, _value in kwargs.items():
                                cmd += f" --{_key} {_value}"

                        cmds.append(cmd)
                        _index += 1

            if len(cmds) > 0:
                gpu_cmds = {}
                for _j in range(self.num_gpus):
                    gpu_cmds[f"gpu_{_j}"] = ""

                for _k in range(len(cmds)):
                    for _j in range(self.num_gpus):
                        if f"--gpu_id {_j}" in cmds[_k]:
                            gpu_cmds[f"gpu_{_j}"] += cmds[_k]
                            gpu_cmds[f"gpu_{_j}"] += "; "
                            break
                gpu_cmds = list(gpu_cmds.values())

                processes = []
                for _i in range(len(gpu_cmds)):
                    gpu_cmd = gpu_cmds[_i]
                    print(f"\n[info] training - stage {_stage + 1}:\n[info] commands: {gpu_cmd}")
                    print(f"[info] log '.txt' inside '{os.path.join(self.nnunet_results, self.dataset_name)}'")
                    processes.append(subprocess.Popen(gpu_cmd, shell=True, stdout=subprocess.DEVNULL))

                for p in processes:
                    p.wait()

    def validate_single_model(self, config, fold, **kwargs):
        """
        Args:
            config: configuration that should be trained.
            fold: fold of the 5-fold cross-validation. Should be an int between 0 and 4.
        """
        self.train_single_model(config=config, fold=fold, only_run_validation=True, **kwargs)

    def validate(self, configs=["3d_fullres", "2d", "3d_lowres", "3d_cascade_fullres"], **kwargs):
        """
        Args:
            configs: configurations that should be trained.
        """
        if isinstance(configs, str):
            configs = [configs]

        for _i in range(len(configs)):
            _config = configs[_i]
            for _fold in range(self.num_folds):
                self.validate_single_model(config=_config, fold=_fold, **kwargs)

    def find_best_configuration(
        self,
        plans="nnUNetPlans",
        configs=["2d", "3d_fullres", "3d_lowres", "3d_cascade_fullres"],
        trainers="nnUNetTrainer",
        allow_ensembling: bool = True,
        num_processes: int = -1,
        overwrite: bool = True,
        folds: Union[List[int], Tuple[int, ...]] = (0, 1, 2, 3, 4),
        strict: bool = False,
    ):
        from nnunetv2.evaluation.find_best_configuration import (
            dumb_trainer_config_plans_to_trained_models_dict,
            find_best_configuration,
        )

        configs = [configs] if isinstance(configs, str) else configs
        plans = [plans] if isinstance(plans, str) else plans
        trainers = [trainers] if isinstance(trainers, str) else trainers

        models = dumb_trainer_config_plans_to_trained_models_dict(trainers, configs, plans)
        num_processes = self.default_num_processes if num_processes < 0 else num_processes
        _ = find_best_configuration(
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
        list_of_lists_or_source_folder: Union[str, List[List[str]]],
        output_folder: str,
        model_training_output_dir: str,
        use_folds: Union[Tuple[int, ...], str] = None,
        tile_step_size: float = 0.5,
        use_gaussian: bool = True,
        use_mirroring: bool = True,
        perform_everything_on_gpu: bool = True,
        verbose: bool = True,
        save_probabilities: bool = False,
        overwrite: bool = True,
        checkpoint_name: str = "checkpoint_final.pth",
        folder_with_segs_from_prev_stage: str = None,
        num_parts: int = 1,
        part_id: int = 0,
        num_processes_preprocessing: int = -1,
        num_processes_segmentation_export: int = -1,
        gpu_id: int = 0,
    ):
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_id}"

        from nnunetv2.inference.predict_from_raw_data import predict_from_raw_data

        n_processes_preprocessing = (
            self.default_num_processes if num_processes_preprocessing < 0 else num_processes_preprocessing
        )
        n_processes_segmentation_export = (
            self.default_num_processes if num_processes_segmentation_export < 0 else num_processes_segmentation_export
        )

        predict_from_raw_data(
            list_of_lists_or_source_folder=list_of_lists_or_source_folder,
            output_folder=output_folder,
            model_training_output_dir=model_training_output_dir,
            use_folds=use_folds,
            tile_step_size=tile_step_size,
            use_gaussian=use_gaussian,
            use_mirroring=use_mirroring,
            perform_everything_on_gpu=perform_everything_on_gpu,
            verbose=verbose,
            save_probabilities=save_probabilities,
            overwrite=overwrite,
            checkpoint_name=checkpoint_name,
            num_processes_preprocessing=n_processes_preprocessing,
            num_processes_segmentation_export=n_processes_segmentation_export,
            folder_with_segs_from_prev_stage=folder_with_segs_from_prev_stage,
            num_parts=num_parts,
            part_id=part_id,
        )

    def predict_ensemble_postprocessing(
        self,
        folds: list = [0, 1, 2, 3, 4],
        disable_ensemble: bool = False,
        disable_predict: bool = False,
        disable_postprocessing: bool = False,
        **kwargs,
    ):
        """
        Args:
            folds: which folds to use
            disable_ensemble: whether to disable ensemble
            disable_predict: whether to predict using trained checkpoints
            disable_postprocessing: whether to conduct post-processing
        """
        from nnunetv2.ensembling.ensemble import ensemble_folders
        from nnunetv2.postprocessing.remove_connected_components import apply_postprocessing_to_folder
        from nnunetv2.utilities.file_path_utilities import get_output_folder

        source_dir = join(self.nnunet_raw, self.dataset_name, "imagesTs")
        target_dir_base = join(self.nnunet_results, self.dataset_name)

        self.best_configuration = ConfigParser.load_config_file(
            os.path.join(self.nnunet_results, self.dataset_name, "inference_information.json")
        )

        if not disable_ensemble:
            has_ensemble = len(self.best_configuration["best_model_or_ensemble"]["selected_model_or_models"]) > 1
        else:
            has_ensemble = False

        used_folds = folds
        output_folders = []
        for im in self.best_configuration["best_model_or_ensemble"]["selected_model_or_models"]:
            output_dir = join(target_dir_base, f"pred_{im['configuration']}")
            output_folders.append(output_dir)

            if not disable_predict:
                model_folder = get_output_folder(
                    int(self.dataset_name_or_id), im["trainer"], im["plans_identifier"], im["configuration"]
                )
                self.predict(
                    list_of_lists_or_source_folder=source_dir,
                    output_folder=output_dir,
                    model_training_output_dir=model_folder,
                    use_folds=used_folds,
                    save_probabilities=has_ensemble,
                    verbose=False,
                    overwrite=True,
                    **kwargs,
                )

        # if we have an ensemble, we need to ensemble the results
        if has_ensemble:
            ensemble_folders(
                output_folders, join(target_dir_base, "ensemble_predictions"), save_merged_probabilities=False
            )
            if not disable_postprocessing:
                folder_for_pp = join(target_dir_base, "ensemble_predictions")
        else:
            if not disable_postprocessing:
                folder_for_pp = output_folders[0]

        # apply postprocessing
        if not disable_postprocessing:
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
        if_convert_dataset: bool = False,
        if_plan_and_process: bool = False,
        if_train: bool = False,
        if_find_best_configuration: bool = False,
        if_predict_ensemble_postprocessing: bool = False,
    ):
        """
        Args:
            if_convert_dataset: whether to convert datasets
            if_plan_and_process: whether to preprocess and analyze the dataset
            if_train: whether to train models
            if_find_best_configuration: whether to find the best model (ensemble) configurations
            if_predict_ensemble_postprocessing: whether to make predictions on test datasets
        """
        if if_convert_dataset:
            self.convert_dataset()

        if if_plan_and_process:
            self.plan_and_process()

        if if_train:
            self.train()

        if if_find_best_configuration:
            self.find_best_configuration()

        if if_predict_ensemble_postprocessing:
            self.predict_ensemble_postprocessing()

        return

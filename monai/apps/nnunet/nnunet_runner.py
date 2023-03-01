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

import nibabel as nib
import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import join, load_pickle

import monai
from monai.bundle import ConfigParser


class nnUNetRunner:
    def __init__(self, input):
        self.input_info = []
        self.input_config_or_dict = input

        if isinstance(self.input_config_or_dict, dict):
            self.input_info = self.input_config_or_dict
        elif isinstance(self.input_config_or_dict, str) and os.path.isfile(self.input_config_or_dict):
            self.input_info = ConfigParser.load_config_file(self.input_config_or_dict)
        else:
            raise ValueError(f"{input} is not a valid file or dict")

        self.nnunet_raw = self.input_info.pop("nnunet_raw", os.path.join(".", "work_dir", "nnUNet_raw_data_base"))
        self.nnunet_preprocessed = self.input_info.pop(
            "nnunet_preprocessed", os.path.join(".", "work_dir", "nnUNet_preprocessed")
        )
        self.nnunet_results = self.input_info.pop(
            "nnunet_results", os.path.join(".", "work_dir", "nnUNet_trained_models")
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

        from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name

        try:
            self.dataset_name = maybe_convert_to_dataset_name(int(self.dataset_name_or_id))
        except:
            print("Dataset ID does not exist! Check input '.yaml' if this is unexpected.")

        from nnunetv2.configuration import default_num_processes

        self.default_num_processes = default_num_processes

        self.num_folds = 5
        self.best_configuration = None

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

            datalist_json = ConfigParser.load_config_file(self.input_info.pop("datalist"))

            if "training" in datalist_json:
                os.makedirs(os.path.join(raw_data_foldername, "imagesTs"))
                os.makedirs(os.path.join(raw_data_foldername, "labelsTs"))
            else:
                print("Input '.json' data list is incorrect.")
                return

            test_key = None
            if "test" in datalist_json or "testing" in datalist_json:
                os.makedirs(os.path.join(raw_data_foldername, "imagesTr"))
                test_key = "test" if "test" in datalist_json else "testing"

            img = monai.transforms.LoadImage(image_only=True, ensure_channel_first=True, simple_keys=True)(
                os.path.join(data_dir, datalist_json["training"][0]["image"])
            )
            num_input_channels = img.size()[0] if img.dim() == 4 else 1
            print(f"num_input_channels: {num_input_channels}")

            num_foreground_classes = 0
            for _i in range(len(datalist_json["training"])):
                seg = monai.transforms.LoadImage(image_only=True, ensure_channel_first=True, simple_keys=True)(
                    os.path.join(data_dir, datalist_json["training"][_i]["label"])
                )
                num_foreground_classes = max(num_foreground_classes, int(seg.max()))
            print(f"num_foreground_classes: {num_foreground_classes}")

            new_json_data = {}

            modality = self.input_info.pop("modality")
            if type(modality) != list:
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

            for _key, _folder in list(zip(["training", test_key], ["imagesTs", "imagesTr"])):
                if _key == None:
                    continue

                for _k in range(len(datalist_json[_key])):
                    orig_img_name = (
                        datalist_json[_key][_k]["image"]
                        if type(datalist_json[_key][_k]) == dict
                        else datalist_json[_key][_k]
                    )
                    img_name = f"image_{_index}"
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
                    if type(datalist_json[_key][_k]) == dict and "label" in datalist_json[_key][_k]:
                        nda = monai.transforms.LoadImage(image_only=True, ensure_channel_first=True, simple_keys=True)(
                            os.path.join(data_dir, datalist_json[_key][_k]["label"])
                        )
                        affine = nda.meta["original_affine"]
                        nda = nda.numpy().astype(np.uint8)
                        nib.save(
                            nib.Nifti1Image(nda, affine),
                            os.path.join(raw_data_foldername, "labelsTs", img_name + ".nii.gz"),
                        )

                    if type(datalist_json[_key][_k]) == dict:
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
        except:
            print("Input '.yaml' is incorrect.")

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
        self, c=["2d", "3d_fullres", "3d_lowres"], np=[8, 4, 8], overwrite_plans_name="nnUNetPlans", verbose=False
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
        np=[8, 4, 8],
        verbose=False,
    ):
        self.extract_fingerprints(fpe, np, verify_dataset_integrity, clean, verbose)
        self.plan_experiments(pl, gpu_memory_target, preprocessor_name, overwrite_target_spacing, overwrite_plans_name)
        self.preprocess(c, np, overwrite_plans_name, verbose)

    def train_single_model(self, config, fold, **kwargs):
        """
        Args:
            config: configuration that should be trained.
            fold: fold of the 5-fold cross-validation. Should be an int between 0 and 4.
            trainer_class_name: name of the custom trainer class. default: 'nnUNetTrainer'.
            plans_identifier: custom plans identifier. default: 'nnUNetPlans'.
            pretrained_weights: path to nnU-Net checkpoint file to be used as pretrained model. Will only be used when actually training. Beta. Use with caution. default: False.
            num_gpus: number of GPUs to use for training. default: 1.
            use_compressed_data: true to use compressed data for training. Reading compressed data is much more CPU and (potentially) RAM intensive and should only be used if you know what you are doing default: False.
            export_validation_probabilities: true to save softmax predictions from final validation as npz files (in addition to predicted segmentations). Needed for finding the best ensemble. default: False.
            continue_training: continue training from latest checkpoint. default: False.
            only_run_validation: true to run the validation only. Requires training to have finished. default: False.
            disable_checkpointing: true to disable checkpointing. Ideal for testing things out and you dont want to flood your hard drive with checkpoints. default: False.
        """
        from nnunetv2.run.run_training import run_training

        run_training(dataset_name_or_id=self.dataset_name_or_id, configuration=config, fold=fold, **kwargs)

    def train(self, configs=["3d_fullres", "2d", "3d_lowres", "3d_cascade_fullres"], **kwargs):
        if type(configs) == str:
            configs = [configs]

        for _i in range(len(configs)):
            _config = configs[_i]
            for _fold in range(self.num_folds):
                self.train_single_model(config=_config, fold=_fold, **kwargs)

    def validate_single_model(self, config, fold):
        self.train_single_model(config=config, fold=fold, only_run_validation=True)

    def validate(self, configs=["3d_fullres", "2d", "3d_lowres", "3d_cascade_fullres"]):
        if type(configs) == str:
            configs = [configs]

        for _i in range(len(configs)):
            _config = configs[_i]
            for _fold in range(self.num_folds):
                self.validate_single_model(config=_config, fold=_fold)

    def find_best_configuration(self):
        from nnunetv2.evaluation.find_best_configuration import (
            dumb_trainer_config_plans_to_trained_models_dict,
            find_best_configuration,
        )

        models = dumb_trainer_config_plans_to_trained_models_dict(
            ["nnUNetTrainer_5epochs"], ["2d", "3d_lowres", "3d_cascade_fullres", "3d_fullres"], ["nnUNetPlans"]
        )
        ret = find_best_configuration(
            int(self.dataset_name_or_id),
            models,
            allow_ensembling=True,
            num_processes=8,
            overwrite=True,
            folds=(0, 1, 2, 3, 4),
            strict=True,
        )
        self.best_configuration = ret

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
    ):
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

    def predict_ensemble(self, folds=[0, 3]):
        from nnunetv2.ensembling.ensemble import ensemble_folders
        from nnunetv2.postprocessing.remove_connected_components import apply_postprocessing_to_folder
        from nnunetv2.utilities.file_path_utilities import get_output_folder

        source_dir = join(self.nnunet_raw, self.dataset_name, "imagesTs")
        target_dir_base = join(self.nnunet_results, self.dataset_name)

        has_ensemble = len(self.best_configuration["best_model_or_ensemble"]["selected_model_or_models"]) > 1

        used_folds = folds
        output_folders = []
        for im in self.best_configuration["best_model_or_ensemble"]["selected_model_or_models"]:
            output_dir = join(target_dir_base, f"pred_{im['configuration']}")
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
            )
            output_folders.append(output_dir)

        # if we have an ensemble, we need to ensemble the results
        if has_ensemble:
            ensemble_folders(
                output_folders, join(target_dir_base, "ensemble_predictions"), save_merged_probabilities=False
            )
            folder_for_pp = join(target_dir_base, "ensemble_predictions")
        else:
            folder_for_pp = output_folders[0]

        # apply postprocessing
        pp_fns, pp_fn_kwargs = load_pickle(self.best_configuration["best_model_or_ensemble"]["postprocessing_file"])
        apply_postprocessing_to_folder(
            folder_for_pp,
            join(target_dir_base, "ensemble_predictions_postprocessed"),
            pp_fns,
            pp_fn_kwargs,
            plans_file_or_dict=self.best_configuration["best_model_or_ensemble"]["some_plans_file"],
        )

    def run(self):
        self.plan_and_process()
        self.train()
        self.find_best_configuration()
        self.predict_ensemble()

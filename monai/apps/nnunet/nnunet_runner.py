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
import yaml

from batchgenerators.utilities.file_and_folder_operations import join, load_pickle


class nnUNetRunner:
    def __init__(self, input):
        self.input_info = []
        self.input_config_or_dict = input
        if isinstance(self.input_config_or_dict, dict):
            self.input_info = self.input_config_or_dict
        elif isinstance(self.input_config_or_dict, str) and os.path.isfile(self.input_config_or_dict):
            with open(self.input_config_or_dict) as f:
                self.input_info = yaml.full_load(f)
        else:
            raise ValueError(f"{input} is not a valid file or dict")

        self.nnunet_raw = self.input_info["nnunet_raw"]
        self.nnunet_preprocessed = self.input_info["nnunet_preprocessed"]
        self.nnunet_results = self.input_info["nnunet_results"]

        # claim environment variable
        os.environ["nnUNet_raw"] = self.nnunet_raw
        os.environ["nnUNet_preprocessed"] = self.nnunet_preprocessed
        os.environ["nnUNet_results"] = self.nnunet_results
        os.environ["OMP_NUM_THREADS"] = str(1)

        # dataset_name_or_id has to be a string
        self.dataset_name_or_id = str(self.input_info["dataset_name_or_id"])

        from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
        self.dataset_name = maybe_convert_to_dataset_name(int(self.dataset_name_or_id))

        from nnunetv2.configuration import default_num_processes
        self.default_num_processes = default_num_processes

        self.num_folds = 5
        self.best_configuration = None

    def convert_dataset(self):
        pass

    def extract_fingerprints(
        self,
        fpe="DatasetFingerprintExtractor",
        npfp=8,
        verify_dataset_integrity=False,
        clean=False,
        verbose=False,
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
        print('Experiment planning...')
        plan_experiments(
            [int(self.dataset_name_or_id)], pl, gpu_memory_target, preprocessor_name, overwrite_target_spacing, overwrite_plans_name
        )

    def preprocess(
        self,
        c=["2d", "3d_fullres", "3d_lowres"],
        np=[8, 4, 8],
        overwrite_plans_name="nnUNetPlans",
        verbose=False,
    ):
        from nnunetv2.experiment_planning.plan_and_preprocess_api import preprocess
        print('Preprocessing...')
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
        from nnunetv2.evaluation.find_best_configuration import find_best_configuration, \
            dumb_trainer_config_plans_to_trained_models_dict

        models = dumb_trainer_config_plans_to_trained_models_dict(
            ["nnUNetTrainer_5epochs"], ["2d", "3d_lowres", "3d_cascade_fullres", "3d_fullres"], ["nnUNetPlans"]
        )
        ret = find_best_configuration(
            int(self.dataset_name_or_id), models, allow_ensembling=True, num_processes=8, overwrite=True, folds=(0, 1, 2, 3, 4), strict=True
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
        checkpoint_name: str = 'checkpoint_final.pth',
        folder_with_segs_from_prev_stage: str = None,
        num_parts: int = 1,
        part_id: int = 0,
    ):
        from nnunetv2.inference.predict_from_raw_data import predict_from_raw_data

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
            num_processes_preprocessing=self.default_num_processes,
            num_processes_segmentation_export=self.default_num_processes,
            folder_with_segs_from_prev_stage=folder_with_segs_from_prev_stage,
            num_parts=num_parts,
            part_id=part_id,
        )

    def predict_ensemble(self, folds=[0, 3]):
        from nnunetv2.ensembling.ensemble import ensemble_folders
        # from nnunetv2.inference.predict_from_raw_data import predict_from_raw_data
        from nnunetv2.postprocessing.remove_connected_components import apply_postprocessing_to_folder
        from nnunetv2.utilities.file_path_utilities import get_output_folder

        source_dir = join(self.nnunet_raw, self.dataset_name, "imagesTs")
        target_dir_base = join(self.nnunet_results, self.dataset_name)

        has_ensemble = len(self.best_configuration["best_model_or_ensemble"]["selected_model_or_models"]) > 1

        used_folds = folds
        output_folders = []
        for im in self.best_configuration["best_model_or_ensemble"]["selected_model_or_models"]:
            output_dir = join(target_dir_base, f"pred_{im['configuration']}")
            model_folder = get_output_folder(int(self.dataset_name_or_id), im["trainer"], im["plans_identifier"], im["configuration"])
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

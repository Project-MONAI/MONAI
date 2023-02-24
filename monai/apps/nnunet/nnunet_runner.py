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


class nnUNetRunner:
    """
    Examples:
        - User can use the one-liner to start the nnU-Net workflow

        .. code-block:: bash

            python -m monai.apps.nnunet nnUNetRunner run --input "./input.yaml"

        - single-gpu training for all 20 models

        .. code-block:: bash

            python -m monai.apps.nnunet nnUNetRunner train --input "./input.yaml"

        - single-gpu training for a single model

        .. code-block:: bash

            python -m monai.apps.nnunet nnUNetRunner train_single_model --input "./input.yaml" \
                --config "3d_fullres" \
                --fold 0 \
                --trainer_class_name "nnUNetTrainer_5epochs" \
                --export_validation_probabilities true

        - multi-gpu training for all 20 models

        .. code-block:: bash

            export CUDA_VISIBLE_DEVICES=0,1 # optional
            python -m monai.apps.nnunet nnUNetRunner train --input "./input.yaml" --num_gpus 2

        - multi-gpu training for a single model

        .. code-block:: bash

            export CUDA_VISIBLE_DEVICES=0,1 # optional
            python -m monai.apps.nnunet nnUNetRunner train_single_model --input "./input.yaml" \
                --config "3d_fullres" \
                --fold 0 \
                --trainer_class_name "nnUNetTrainer_5epochs" \
                --export_validation_probabilities true \
                --num_gpus 2
    """
    def __init__(self, input):
        self.input = input
        with open(self.input) as f:
            input_info = yaml.full_load(f)

        # claim environment variable
        os.environ["nnUNet_raw"] = input_info["nnunet_raw"]
        os.environ["nnUNet_preprocessed"] = input_info["nnunet_preprocessed"]
        os.environ["nnUNet_results"] = input_info["nnunet_results"]
        os.environ["OMP_NUM_THREADS"] = str(1)

        # dataset_name_or_id has to be a string
        self.dataset_name_or_id = str(input_info["dataset_name_or_id"])

        self.num_folds = 5

    def plan_and_process(self):
        pass

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
        dataset_name = maybe_convert_to_dataset_name(d)
        source_dir = join(nnUNet_raw, dataset_name, "imagesTs")
        target_dir_base = join(nnUNet_results, dataset_name)

        models = dumb_trainer_config_plans_to_trained_models_dict(
            ["nnUNetTrainer_5epochs"], ["2d", "3d_lowres", "3d_cascade_fullres", "3d_fullres"], ["nnUNetPlans"]
        )
        ret = find_best_configuration(
            d, models, allow_ensembling=True, num_processes=8, overwrite=True, folds=(0, 1, 2, 3, 4), strict=True
        )

    def ensemble(self):
        has_ensemble = len(ret["best_model_or_ensemble"]["selected_model_or_models"]) > 1

        # we don't use all folds to speed stuff up
        used_folds = (0, 3)
        output_folders = []
        for im in ret["best_model_or_ensemble"]["selected_model_or_models"]:
            output_dir = join(target_dir_base, f"pred_{im['configuration']}")
            model_folder = get_output_folder(d, im["trainer"], im["plans_identifier"], im["configuration"])
            # note that if the best model is the enseble of 3d_lowres and 3d cascade then 3d_lowres will be predicted
            # twice (once standalone and once to generate the predictions for the cascade) because we don't reuse the
            # prediction here. Proper way would be to check for that and
            # then give the output of 3d_lowres inference to the folder_with_segs_from_prev_stage kwarg in
            # predict_from_raw_data. Since we allow for
            # dynamically setting 'previous_stage' in the plans I am too lazy to implement this here. This is just an
            # integration test after all. Take a closer look at how this in handled in predict_from_raw_data
            predict_from_raw_data(
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
        pp_fns, pp_fn_kwargs = load_pickle(ret["best_model_or_ensemble"]["postprocessing_file"])
        apply_postprocessing_to_folder(
            folder_for_pp,
            join(target_dir_base, "ensemble_predictions_postprocessed"),
            pp_fns,
            pp_fn_kwargs,
            plans_file_or_dict=ret["best_model_or_ensemble"]["some_plans_file"],
        )

    def run(self):
        self.plan_and_process()
        self.train()
        self.find_best_configuration()
        self.ensemble()

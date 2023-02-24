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
import subprocess

import yaml


class nnUNetRunner:
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

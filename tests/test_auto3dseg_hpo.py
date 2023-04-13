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
import tempfile
import unittest
from functools import partial

import torch

from monai.apps.auto3dseg import BundleGen, DataAnalyzer, NNIGen, OptunaGen, import_bundle_algo_history
from monai.bundle.config_parser import ConfigParser
from monai.utils import optional_import
from monai.utils.enums import AlgoKeys
from tests.utils import (
    SkipIfBeforePyTorchVersion,
    export_fake_data_config_file,
    generate_fake_segmentation_data,
    get_testing_algo_template_path,
    skip_if_downloading_fails,
    skip_if_no_cuda,
)

_, has_tb = optional_import("torch.utils.tensorboard", name="SummaryWriter")
optuna, has_optuna = optional_import("optuna")

override_param = (
    {
        "num_images_per_batch": 2,
        "num_epochs": 2,
        "num_epochs_per_validation": 1,
        "num_warmup_epochs": 1,
        "use_pretrain": False,
        "pretrained_path": "",
    }
    if torch.cuda.is_available()
    else {}
)


def skip_if_no_optuna(obj):
    """
    Skip the unit tests if torch.cuda.is_available is False.
    """
    return unittest.skipUnless(has_optuna, "Skipping optuna tests")(obj)


@SkipIfBeforePyTorchVersion((1, 9, 1))
@unittest.skipIf(not has_tb, "no tensorboard summary writer")
class TestHPO(unittest.TestCase):
    def setUp(self) -> None:
        self.test_dir = tempfile.TemporaryDirectory()
        test_path = self.test_dir.name

        work_dir = os.path.abspath(os.path.join(test_path, "workdir"))
        dataroot = os.path.join(work_dir, "dataroot")

        da_output_yaml = os.path.join(work_dir, "datastats.yaml")
        data_src_cfg = os.path.join(work_dir, "data_src_cfg.yaml")

        if not os.path.isdir(dataroot):
            os.makedirs(dataroot)

        if not os.path.isdir(work_dir):
            os.makedirs(work_dir)

        # Generate a fake dataset
        generate_fake_segmentation_data(dataroot)

        # write to a json file
        fake_json_datalist = export_fake_data_config_file(dataroot)

        da = DataAnalyzer(fake_json_datalist, dataroot, output_path=da_output_yaml)
        da.get_all_case_stats()

        data_src = {
            "name": "fake_data",
            "task": "segmentation",
            "modality": "MRI",
            "datalist": fake_json_datalist,
            "dataroot": dataroot,
            "multigpu": False,
            "class_names": ["label_class"],
        }

        ConfigParser.export_config_file(data_src, data_src_cfg)
        with skip_if_downloading_fails():
            bundle_generator = BundleGen(
                algo_path=work_dir,
                data_stats_filename=da_output_yaml,
                data_src_cfg_name=data_src_cfg,
                templates_path_or_url=get_testing_algo_template_path(),
            )
        bundle_generator.generate(work_dir, num_fold=1)

        self.history = bundle_generator.get_history()
        self.work_dir = work_dir
        self.test_path = test_path

    @skip_if_no_cuda
    def test_run_algo(self) -> None:
        algo_dict = self.history[-1]
        algo = algo_dict[AlgoKeys.ALGO]
        nni_gen = NNIGen(algo=algo, params=override_param)
        obj_filename = nni_gen.get_obj_filename()
        # this function will be used in HPO via Python Fire
        NNIGen().run_algo(obj_filename, self.work_dir)

    @skip_if_no_cuda
    @skip_if_no_optuna
    def test_run_optuna(self) -> None:
        algo_dict = self.history[-1]
        algo = algo_dict[AlgoKeys.ALGO]

        class OptunaGenLearningRate(OptunaGen):
            def get_hyperparameters(self):
                return {"learning_rate": self.trial.suggest_float("learning_rate", 0.00001, 0.1)}

        optuna_gen = OptunaGenLearningRate(algo=algo, params=override_param)
        search_space = {"learning_rate": [0.0001, 0.001, 0.01, 0.1]}
        study = optuna.create_study(sampler=optuna.samplers.GridSampler(search_space), direction="maximize")
        study.optimize(
            partial(
                optuna_gen,
                obj_filename=optuna_gen.get_obj_filename(),
                output_folder=os.path.join(self.test_path, "optuna_test"),
            ),
            n_trials=2,
        )
        print(f"Best value: {study.best_value} (params: {study.best_params})\n")

    @skip_if_no_cuda
    def test_get_history(self) -> None:
        algo_dict = self.history[-1]
        algo = algo_dict[AlgoKeys.ALGO]
        nni_gen = NNIGen(algo=algo, params=override_param)
        obj_filename = nni_gen.get_obj_filename()

        NNIGen().run_algo(obj_filename, self.work_dir)

        history = import_bundle_algo_history(self.work_dir, only_trained=True)
        assert len(history) == 1

    def tearDown(self) -> None:
        self.test_dir.cleanup()


if __name__ == "__main__":
    unittest.main()

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

import os
import shutil
import tempfile
import unittest
from functools import partial
from typing import Dict, List

import nibabel as nib
import numpy as np

from monai.apps.auto3dseg import BundleGen, DataAnalyzer, NNIGen, OptunaGen, import_bundle_algo_history
from monai.bundle.config_parser import ConfigParser
from monai.data import create_test_image_3d
from monai.utils import optional_import
from tests.utils import SkipIfBeforePyTorchVersion, skip_if_downloading_fails, skip_if_no_cuda

_, has_tb = optional_import("torch.utils.tensorboard", name="SummaryWriter")
optuna, has_optuna = optional_import("optuna")


def skip_if_no_optuna(obj):
    """
    Skip the unit tests if torch.cuda.is_available is False.
    """
    return unittest.skipUnless(has_optuna, "Skipping optuna tests")(obj)


fake_datalist: Dict[str, List[Dict]] = {
    "testing": [{"image": "val_001.fake.nii.gz"}, {"image": "val_002.fake.nii.gz"}],
    "training": [
        {"fold": 0, "image": "tr_image_001.fake.nii.gz", "label": "tr_label_001.fake.nii.gz"},
        {"fold": 0, "image": "tr_image_002.fake.nii.gz", "label": "tr_label_002.fake.nii.gz"},
        {"fold": 0, "image": "tr_image_003.fake.nii.gz", "label": "tr_label_003.fake.nii.gz"},
        {"fold": 0, "image": "tr_image_004.fake.nii.gz", "label": "tr_label_004.fake.nii.gz"},
        {"fold": 1, "image": "tr_image_005.fake.nii.gz", "label": "tr_label_005.fake.nii.gz"},
        {"fold": 1, "image": "tr_image_006.fake.nii.gz", "label": "tr_label_006.fake.nii.gz"},
        {"fold": 1, "image": "tr_image_007.fake.nii.gz", "label": "tr_label_007.fake.nii.gz"},
        {"fold": 1, "image": "tr_image_008.fake.nii.gz", "label": "tr_label_008.fake.nii.gz"},
        {"fold": 2, "image": "tr_image_009.fake.nii.gz", "label": "tr_label_009.fake.nii.gz"},
        {"fold": 2, "image": "tr_image_010.fake.nii.gz", "label": "tr_label_010.fake.nii.gz"},
        {"fold": 2, "image": "tr_image_011.fake.nii.gz", "label": "tr_label_011.fake.nii.gz"},
        {"fold": 2, "image": "tr_image_012.fake.nii.gz", "label": "tr_label_012.fake.nii.gz"},
    ],
}


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
        for d in fake_datalist["testing"] + fake_datalist["training"]:
            im, seg = create_test_image_3d(64, 64, 64, rad_max=10, num_seg_classes=1)
            nib_image = nib.Nifti1Image(im, affine=np.eye(4))
            image_fpath = os.path.join(dataroot, d["image"])
            nib.save(nib_image, image_fpath)

            if "label" in d:
                nib_image = nib.Nifti1Image(seg, affine=np.eye(4))
                label_fpath = os.path.join(dataroot, d["label"])
                nib.save(nib_image, label_fpath)

        # write to a json file
        fake_json_datalist = os.path.join(dataroot, "fake_input.json")
        ConfigParser.export_config_file(fake_datalist, fake_json_datalist)

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
                algo_path=work_dir, data_stats_filename=da_output_yaml, data_src_cfg_name=data_src_cfg
            )
        bundle_generator.generate(work_dir, num_fold=2)

        self.history = bundle_generator.get_history()
        self.work_dir = work_dir
        self.test_path = test_path

    @skip_if_no_cuda
    def test_run_algo(self) -> None:
        override_param = {
            "num_iterations": 8,
            "num_iterations_per_validation": 4,
            "num_images_per_batch": 2,
            "num_epochs": 2,
            "num_warmup_iterations": 4,
        }

        algo_dict = self.history[0]
        algo_name = list(algo_dict.keys())[0]
        algo = algo_dict[algo_name]
        nni_gen = NNIGen(algo=algo, params=override_param)
        obj_filename = nni_gen.get_obj_filename()
        # this function will be used in HPO via Python Fire
        NNIGen().run_algo(obj_filename, self.work_dir)

    @skip_if_no_cuda
    @skip_if_no_optuna
    def test_run_optuna(self) -> None:
        override_param = {
            "num_iterations": 8,
            "num_iterations_per_validation": 4,
            "num_images_per_batch": 2,
            "num_epochs": 2,
            "num_warmup_iterations": 4,
        }

        algo_dict = self.history[0]
        algo_name = list(algo_dict.keys())[0]
        algo = algo_dict[algo_name]

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
    def test_run_algo_after_move_files(self) -> None:
        override_param = {
            "num_iterations": 8,
            "num_iterations_per_validation": 4,
            "num_images_per_batch": 2,
            "num_epochs": 2,
            "num_warmup_iterations": 4,
        }

        algo_dict = self.history[0]
        algo_name = list(algo_dict.keys())[0]
        algo = algo_dict[algo_name]
        nni_gen = NNIGen(algo=algo, params=override_param)
        obj_filename = nni_gen.get_obj_filename()

        work_dir_2 = os.path.join(self.test_path, "workdir2")
        os.makedirs(work_dir_2)
        algorithm_template = os.path.join(self.work_dir, "algorithm_templates")
        algorithm_templates_2 = os.path.join(work_dir_2, "algorithm_templates")
        algo_dir = os.path.dirname(obj_filename)
        algo_dir_2 = os.path.join(work_dir_2, os.path.basename(algo_dir))

        obj_filename_2 = os.path.join(algo_dir_2, "algo_object.pkl")
        shutil.copytree(algorithm_template, algorithm_templates_2)
        shutil.copytree(algo_dir, algo_dir_2)
        # this function will be used in HPO via Python Fire in remote
        NNIGen().run_algo(obj_filename_2, work_dir_2, template_path=algorithm_templates_2)

    @skip_if_no_cuda
    def test_get_history(self) -> None:
        override_param = {
            "num_iterations": 8,
            "num_iterations_per_validation": 4,
            "num_images_per_batch": 2,
            "num_epochs": 2,
            "num_warmup_iterations": 4,
        }

        algo_dict = self.history[0]
        algo_name = list(algo_dict.keys())[0]
        algo = algo_dict[algo_name]
        nni_gen = NNIGen(algo=algo, params=override_param)
        obj_filename = nni_gen.get_obj_filename()

        NNIGen().run_algo(obj_filename, self.work_dir)

        history = import_bundle_algo_history(self.work_dir, only_trained=True)
        assert len(history) == 1

    def tearDown(self) -> None:
        self.test_dir.cleanup()


if __name__ == "__main__":
    unittest.main()

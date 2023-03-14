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

from monai.apps.auto3dseg import AutoRunner
from monai.bundle.config_parser import ConfigParser
from monai.utils import optional_import
from tests.utils import export_fake_data_config_file, generate_fake_segmentation_data, skip_if_quick

health_azure, has_health_azure = optional_import("health_azure")


@skip_if_quick
@unittest.skipIf(not has_health_azure, "health_azure package is required for this test.")
class TestAuto3DSegAzureML(unittest.TestCase):
    def setUp(self) -> None:
        self.test_dir = tempfile.TemporaryDirectory(dir=".")
        self.dataroot = os.path.join(self.test_dir.name, "data")
        os.makedirs(self.dataroot)

    def test_submit_autorunner_job_to_azureml(self) -> None:
        # generate fake data and datalist
        generate_fake_segmentation_data(self.dataroot)
        fake_json_datalist = export_fake_data_config_file(self.dataroot)

        # set up test task yaml
        self.test_task_yaml = os.path.join(self.test_dir.name, "fake_task.yaml")
        azureml_test_task_src = {
            "name": "test_task",
            "task": "segmentation",
            "modality": "MRI",
            "datalist": fake_json_datalist,
            "dataroot": self.dataroot,
            "multigpu": True,
            "azureml_config": {
                "compute_cluster_name": "dedicated-nc24s-v2",
                "default_datastore": "himldatasets",
                "wait_for_completion": True,
            },
        }
        ConfigParser.export_config_file(azureml_test_task_src, self.test_task_yaml)

        # set up test training params
        test_params = {
            "num_epochs_per_validation": 1,
            "num_images_per_batch": 1,
            "num_epochs": 2,
            "num_warmup_epochs": 1,
        }
        test_num_fold = 2

        # run AutoRunner in AzureML
        with self.assertRaises(SystemExit) as cm:
            with unittest.mock.patch(
                "sys.argv",
                [
                    "__main__.py",
                    "AutoRunner",
                    "run",
                    f"--input={self.test_task_yaml}",
                    f"--training_params={test_params}",
                    f"--num_fold={test_num_fold}",
                    "--azureml",
                ],
            ):
                AutoRunner(input=self.test_task_yaml, training_params=test_params, num_fold=test_num_fold)

        self.assertEqual(cm.exception.code, 0)

    def tearDown(self) -> None:
        self.test_dir.cleanup()


if __name__ == "__main__":
    unittest.main()

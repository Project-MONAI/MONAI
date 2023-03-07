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

import unittest

from monai.apps.auto3dseg import AutoRunner
from monai.utils import optional_import
from tests.utils import skip_if_quick

health_azure, has_health_azure = optional_import("health_azure")


@skip_if_quick
class TestAuto3DSegAzureML(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()

    def test_submit_autorunner_job_to_azureml(self) -> None:
        test_input = './azureml_configs/auto3dseg_test_task.yaml'
        test_params = {
            "num_epochs_per_validation": 1,
            "num_images_per_batch": 1,
            "num_epochs": 2,
            "num_warmup_epochs": 1
        }
        test_num_fold = 2
        with self.assertRaises(SystemExit) as cm:
            with unittest.mock.patch('sys.argv', [
                '__main__.py',
                'AutoRunner',
                'run',
                f'--input={test_input}',
                f'--training_params={test_params}',
                f'--num_fold={test_num_fold}',
                '--azureml'
            ]):
                AutoRunner(input=test_input, training_params=test_params, num_fold=test_num_fold)

        self.assertEqual(cm.exception.code, 0)


if __name__ == "__main__":
    unittest.main()

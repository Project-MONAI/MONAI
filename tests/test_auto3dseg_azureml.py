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
from monai.utils import optional_import
from tests.utils import generate_fake_segmentation_data, skip_if_quick

health_azure, has_health_azure = optional_import("health_azure")

fake_datalist: dict[str, list[dict]] = {
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


@skip_if_quick
@unittest.skipIf(not has_health_azure, "health_azure package is required for this test.")
class TestAuto3DSegAzureML(unittest.TestCase):
    def setUp(self) -> None:
        self.test_dir = tempfile.TemporaryDirectory(dir=".")
        os.makedirs(os.path.join(self.test_dir.name, "data"))

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

    def tearDown(self) -> None:
        self.test_dir.cleanup()


if __name__ == "__main__":
    unittest.main()

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
import unittest

import torch
from parameterized import parameterized

from monai.apps.mmars import MODEL_DESC, load_from_mmar
from monai.config import print_debug_info


class TestAllDownloadingMMAR(unittest.TestCase):
    def setUp(self):
        print_debug_info()
        self.test_dir = "./"

    @parameterized.expand((item,) for item in MODEL_DESC)
    def test_loading_mmar(self, item):
        if item["name"] == "clara_pt_fed_learning_brain_tumor_mri_segmentation":
            default_model_file = os.path.join("models", "server", "best_FL_global_model.pt")
        else:
            default_model_file = None
        pretrained_model = load_from_mmar(
            item=item["name"], mmar_dir="./", map_location="cpu", api=True, model_file=default_model_file
        )
        self.assertTrue(isinstance(pretrained_model, torch.nn.Module))

    def tearDown(self):
        print(os.listdir(self.test_dir))


if __name__ == "__main__":
    unittest.main()

# Copyright 2020 - 2021 MONAI Consortium
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
import tempfile
import unittest

import torch

from monai.networks import save_to_torchscript
from monai.networks.nets import UNet


class TestArrayDataset(unittest.TestCase):
    def test_shape(self):
        model = UNet(
            spatial_dims=2, in_channels=1, out_channels=3, channels=(16, 32, 64), strides=(2, 2), num_res_units=0
        )
        with tempfile.TemporaryDirectory() as tempdir:
            torchscript_model = save_to_torchscript(
                model=model,
                output_path=os.path.join(tempdir, "model.ts"),
                verify=True,
                input_shape=(16, 1, 32, 32),
                device="cuda" if torch.cuda.is_available() else "cpu",
                rtol=1e-3,
            )
            self.assertTrue(isinstance(torchscript_model, torch.jit._script.RecursiveScriptModule))


if __name__ == "__main__":
    unittest.main()

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

import torch

from monai.networks import convert_to_export
from monai.networks.nets import UNet
from monai.utils.module import pytorch_after


class TestConvertToExport(unittest.TestCase):
    def test_basic_export(self):
        """Export a UNet and verify output matches."""
        model = UNet(
            spatial_dims=2, in_channels=1, out_channels=3, channels=(16, 32, 64), strides=(2, 2), num_res_units=0
        )
        with tempfile.TemporaryDirectory() as tempdir:
            exported = convert_to_export(
                model=model,
                filename_or_obj=os.path.join(tempdir, "model.pt2"),
                verify=True,
                inputs=[torch.randn((16, 1, 32, 32), requires_grad=False)],
                device="cpu",
                rtol=1e-3,
                atol=1e-4,
            )
            self.assertIsInstance(exported, torch.export.ExportedProgram)

    def test_export_without_save(self):
        """Export a model without saving to disk."""
        model = UNet(
            spatial_dims=2, in_channels=1, out_channels=3, channels=(16, 32, 64), strides=(2, 2), num_res_units=0
        )
        exported = convert_to_export(model=model, inputs=[torch.randn((2, 1, 32, 32))])
        self.assertIsInstance(exported, torch.export.ExportedProgram)
        out = exported.module()(torch.randn(2, 1, 32, 32))
        self.assertEqual(out.shape, torch.Size([2, 3, 32, 32]))

    def test_missing_inputs_raises(self):
        """Verify that missing inputs raise ValueError."""
        model = UNet(
            spatial_dims=2, in_channels=1, out_channels=3, channels=(16, 32, 64), strides=(2, 2), num_res_units=0
        )
        with self.assertRaises(ValueError):
            convert_to_export(model=model)

    @unittest.skipUnless(pytorch_after(2, 9), "torch.export.Dim.DYNAMIC requires PyTorch >= 2.9")
    def test_export_with_dynamic_shapes(self):
        """Export with dynamic batch dimension."""
        model = UNet(
            spatial_dims=2, in_channels=1, out_channels=3, channels=(16, 32, 64), strides=(2, 2), num_res_units=0
        )
        dynamic = torch.export.Dim.DYNAMIC
        static = torch.export.Dim.STATIC
        exported = convert_to_export(
            model=model, inputs=[torch.randn((2, 1, 32, 32))], dynamic_shapes=((dynamic, static, dynamic, dynamic),)
        )
        # Verify works with different batch size and spatial dims
        out = exported.module()(torch.randn(4, 1, 64, 64))
        self.assertEqual(out.shape, torch.Size([4, 3, 64, 64]))


if __name__ == "__main__":
    unittest.main()

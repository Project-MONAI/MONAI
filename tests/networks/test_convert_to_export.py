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
from unittest.mock import patch

import torch
from parameterized import parameterized

from monai.networks import convert_to_export
from monai.networks.nets import UNet
from monai.utils.module import pytorch_after
from tests.test_utils import skip_if_no_cuda


class _Wrapped(torch.nn.Module):
    def __init__(self, wrap):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 1)
        self.wrap = wrap

    def forward(self, x):
        return self.wrap(self.conv(x))


class _KeepsLastOutput(torch.nn.Module):
    """Keeps its last training output, a tensor that cannot be deep-copied, as some networks do."""

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 1)
        self.last = None

    def forward(self, x):
        out = self.conv(x)
        if self.training:
            self.last = out
        return out


class _AddsOnInputDevice(torch.nn.Module):
    """Creates a tensor on the input's device, which the exported graph keeps as the export device."""

    def forward(self, x):
        return x + torch.ones(x.shape, device=x.device)


class _TakesDict(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)

    def forward(self, inputs):
        return self.linear(inputs["a"])


class _TakesKeyword(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)

    def forward(self, x, y):
        return self.linear(x) + y


def _build_unet():
    return UNet(spatial_dims=2, in_channels=1, out_channels=3, channels=(16, 32, 64), strides=(2, 2), num_res_units=0)


class TestConvertToExport(unittest.TestCase):
    def test_basic_export(self):
        """Export a UNet and verify output matches."""
        model = _build_unet()
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

    def test_verify_checks_saved_file(self):
        """Verification must use the saved file, so a saved model with other weights fails."""
        model = _build_unet()
        inputs = [torch.randn((2, 1, 32, 32))]
        other = torch.export.export(_build_unet().eval(), args=tuple(inputs))
        with tempfile.TemporaryDirectory() as tempdir:
            with patch("torch.export.load", return_value=other) as load:
                with self.assertRaises(AssertionError):
                    convert_to_export(
                        model=model,
                        filename_or_obj=os.path.join(tempdir, "model.pt2"),
                        verify=True,
                        inputs=inputs,
                        device="cpu",
                    )
        load.assert_called_once()

    @parameterized.expand([[lambda x: {"out": x}], [lambda x: ([x],)]])
    def test_verify_compares_nested_outputs(self, wrap):
        """Verification must reach tensors inside dicts and nested sequences."""
        inputs = [torch.randn((2, 1, 8, 8))]
        other = torch.export.export(_Wrapped(wrap).eval(), args=tuple(inputs))
        with tempfile.TemporaryDirectory() as tempdir:
            with patch("torch.export.load", return_value=other):
                with self.assertRaises(AssertionError):
                    convert_to_export(
                        _Wrapped(wrap), os.path.join(tempdir, "model.pt2"), verify=True, inputs=inputs, device="cpu"
                    )

    def test_verify_model_after_training_step(self):
        """Verification must not need to copy the model, which can hold tensors from earlier training steps."""
        net = _KeepsLastOutput()
        data = torch.randn(2, 1, 8, 8)
        net(data).sum().backward()
        convert_to_export(net, verify=True, inputs=[data], device="cpu")

    @skip_if_no_cuda
    def test_verify_leaves_devices_alone(self):
        """Verifying on another device moves neither the model nor the returned program."""
        model = _build_unet()
        exported = convert_to_export(model, verify=True, inputs=[torch.randn((2, 1, 32, 32))], device="cuda:0")
        self.assertEqual(next(model.parameters()).device.type, "cpu")
        self.assertEqual(next(iter(exported.state_dict.values())).device.type, "cpu")

    def test_verify_default_device_is_model_device(self):
        """Without ``device``, verification runs where the model is, so a device kept in the graph still matches."""
        convert_to_export(_AddsOnInputDevice(), verify=True, inputs=[torch.randn(2, 3)])

    @skip_if_no_cuda
    def test_verify_moves_nested_inputs(self):
        """Tensors inside dict inputs are moved to the verification device."""
        convert_to_export(_TakesDict(), verify=True, inputs=[{"a": torch.randn(2, 3)}], device="cuda:0")

    def test_verify_with_keyword_inputs(self):
        """Keyword inputs passed to export through ``kwargs`` are also used for verification."""
        convert_to_export(
            _TakesKeyword(), verify=True, inputs=[torch.randn(2, 3)], kwargs={"y": torch.randn(2, 3)}, device="cpu"
        )

    def test_extra_files(self):
        """Extra files are stored in the saved archive."""
        with tempfile.TemporaryDirectory() as tempdir:
            filename = os.path.join(tempdir, "model.pt2")
            convert_to_export(
                _build_unet(), filename, extra_files={"a.txt": "data"}, inputs=[torch.randn((2, 1, 32, 32))]
            )
            loaded = {"a.txt": ""}
            torch.export.load(filename, extra_files=loaded)
        self.assertEqual(loaded["a.txt"], "data")

    def test_export_without_save(self):
        """Export a model without saving to disk."""
        model = _build_unet()
        exported = convert_to_export(model=model, inputs=[torch.randn((2, 1, 32, 32))])
        self.assertIsInstance(exported, torch.export.ExportedProgram)
        out = exported.module()(torch.randn(2, 1, 32, 32))
        self.assertEqual(out.shape, torch.Size([2, 3, 32, 32]))

    def test_missing_inputs_raises(self):
        """Verify that missing inputs raise ValueError."""
        model = _build_unet()
        with self.assertRaises(ValueError):
            convert_to_export(model=model)

    @unittest.skipUnless(pytorch_after(2, 9), "torch.export.Dim.DYNAMIC requires PyTorch >= 2.9")
    def test_export_with_dynamic_shapes(self):
        """Export with dynamic batch dimension."""
        model = _build_unet()
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

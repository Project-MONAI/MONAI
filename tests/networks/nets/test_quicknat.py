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

import torch
from parameterized import parameterized

from monai.networks import eval_mode
from monai.networks.nets import Quicknat
from monai.networks.nets.quicknat import SkipConnectionWithIdx
from monai.utils import optional_import
from tests.test_utils import test_script_save

_, has_se = optional_import("squeeze_and_excitation")


class _DoubleWithIdx(torch.nn.Module):
    """Double tensor values using QuickNAT's indexed-module signature."""

    def __init__(self):
        super().__init__()
        self.received_indices = None

    def forward(self, tensor, indices):
        """Return doubled values and the unchanged index object.

        Args:
            tensor: input tensor to double.
            indices: pooling indices passed through the indexed module.

        Returns:
            A tuple containing the doubled tensor and the original indices.
        """
        self.received_indices = indices
        return tensor * 2, indices


TEST_CASES = [
    # params, input_shape, expected_shape
    [{"num_classes": 1, "num_channels": 1, "num_filters": 1, "se_block": None}, (1, 1, 32, 32), (1, 1, 32, 32)],
    [{"num_classes": 1, "num_channels": 1, "num_filters": 4, "se_block": None}, (1, 1, 64, 64), (1, 1, 64, 64)],
    [{"num_classes": 1, "num_channels": 1, "num_filters": 64, "se_block": None}, (1, 1, 128, 128), (1, 1, 128, 128)],
    [{"num_classes": 4, "num_channels": 1, "num_filters": 64, "se_block": None}, (1, 1, 32, 32), (1, 4, 32, 32)],
    [{"num_classes": 33, "num_channels": 1, "num_filters": 64, "se_block": None}, (1, 1, 32, 32), (1, 33, 32, 32)],
    [{"num_classes": 1, "num_channels": 1, "num_filters": 64, "se_block": "CSE"}, (1, 1, 32, 32), (1, 1, 32, 32)],
    [{"num_classes": 1, "num_channels": 1, "num_filters": 64, "se_block": "SSE"}, (1, 1, 32, 32), (1, 1, 32, 32)],
    [{"num_classes": 1, "num_channels": 1, "num_filters": 64, "se_block": "CSSE"}, (1, 1, 32, 32), (1, 1, 32, 32)],
]


class TestQuicknatCore(unittest.TestCase):
    """Test QuickNAT paths that do not require optional dependencies."""

    @parameterized.expand(["cat", "add", "mul"])
    def test_skip_connection_modes_preserve_indices(self, mode):
        """Verify each fusion mode and preservation of pooling indices."""
        submodule = _DoubleWithIdx()
        skip = SkipConnectionWithIdx(submodule, mode=mode)
        tensor = torch.tensor([[[[2.0]]]])
        indices = torch.tensor([[[[3]]]])

        output, returned_indices = skip(tensor, indices)

        expected = {"cat": torch.cat([tensor, tensor * 2], dim=1), "add": tensor * 3, "mul": tensor * tensor * 2}[mode]
        self.assertTrue(torch.equal(output, expected))
        self.assertIs(submodule.received_indices, indices)
        self.assertIs(returned_indices, indices)

    def test_skip_connection_rejects_unsupported_mode(self):
        """Verify unsupported fusion modes fail explicitly."""
        skip = SkipConnectionWithIdx(_DoubleWithIdx(), mode="cat")
        skip.mode = "unsupported"

        with self.assertRaisesRegex(NotImplementedError, "Unsupported mode"):
            skip(torch.ones((1, 1, 1, 1)), torch.ones((1, 1, 1, 1)))

    def test_forward_without_optional_se_dependency(self):
        """Verify QuickNAT runs when squeeze-and-excitation is disabled."""
        net = Quicknat(num_classes=2, num_channels=1, num_filters=4, se_block=None)
        with eval_mode(net):
            result = net(torch.randn(1, 1, 32, 32))
        self.assertEqual(result.shape, (1, 2, 32, 32))


@unittest.skipUnless(has_se, "squeeze_and_excitation not installed")
class TestQuicknat(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_shape(self, input_param, input_shape, expected_shape):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(input_param)
        net = Quicknat(**input_param).to(device)
        with eval_mode(net):
            result = net(torch.randn(input_shape).to(device))
        self.assertEqual(result.shape, expected_shape)

    def test_script(self):
        net = Quicknat(num_classes=1, num_channels=1)
        test_data = torch.randn(16, 1, 32, 32)
        test_script_save(net, test_data)


if __name__ == "__main__":
    unittest.main()

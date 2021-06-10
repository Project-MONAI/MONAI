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

import unittest

import numpy as np
import torch
from parameterized import parameterized

from monai.networks.utils import compatible_mod_state
from monai.utils import set_determinism


class _TestModelOne(torch.nn.Module):
    def __init__(self, n_n, n_m, n_class):
        super(_TestModelOne, self).__init__()
        self.layer = torch.nn.Linear(n_n, n_m)
        self.class_layer = torch.nn.Linear(n_m, n_class)

    def forward(self, x):
        x = self.layer(x)
        x = self.class_layer(x)
        return x


class _TestModelTwo(torch.nn.Module):
    def __init__(self, n_n, n_m, n_d, n_class):
        super(_TestModelTwo, self).__init__()
        self.layer = torch.nn.Linear(n_n, n_m)
        self.layer_1 = torch.nn.Linear(n_m, n_d)
        self.class_layer = torch.nn.Linear(n_d, n_class)

    def forward(self, x):
        x = self.layer(x)
        x = self.layer_1(x)
        x = self.class_layer(x)
        return x


TEST_CASES = []
__devices = ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",)
for x in __devices:
    for y in __devices:
        TEST_CASES.append((x, y))


class TestModuleState(unittest.TestCase):
    def setUp(self):
        set_determinism(0)

    def tearDown(self):
        set_determinism(None)

    @parameterized.expand(TEST_CASES)
    def test_set_state(self, device_0, device_1):
        model_one = _TestModelOne(10, 20, 3)
        model_two = _TestModelTwo(10, 20, 10, 4)
        model_one.to(device_0)
        model_two.to(device_1)
        model_dict, ch, unch = compatible_mod_state(model_one, model_two)
        model_one.load_state_dict(model_dict)
        x = torch.randn((4, 10), device=device_0)
        output = model_one(x).detach().cpu().numpy()
        expected = np.array(
            [
                [0.3876864, -0.10779603, -0.35779887],
                [-0.6996534, -0.17674239, -0.48923326],
                [0.06991734, -0.45093504, -0.24457899],
                [0.24552809, -0.41014993, 0.19142818],
            ]
        )
        np.testing.assert_allclose(output, expected, atol=1e-3)
        self.assertEqual(len(ch), 2)
        self.assertEqual(len(unch), 2)

    @parameterized.expand(TEST_CASES)
    def test_set_full_state(self, device_0, device_1):
        model_one = _TestModelOne(10, 20, 3)
        model_two = _TestModelOne(10, 20, 3)
        model_one.to(device_0)
        model_two.to(device_1)
        # test module input
        model_dict, ch, unch = compatible_mod_state(model_one, model_two)
        # test dict input
        model_dict, ch, unch = compatible_mod_state(model_dict, model_two)
        model_one.load_state_dict(model_dict)
        x = torch.randn((4, 10), device=device_0)
        output = model_one(x).detach().cpu().numpy()
        model_two.to(device_0)
        output_1 = model_two(x).detach().cpu().numpy()
        np.testing.assert_allclose(output, output_1, atol=1e-3)
        self.assertEqual(len(ch), 4)
        self.assertEqual(len(unch), 0)

    @parameterized.expand(TEST_CASES)
    def test_set_map_state(self, device_0, device_1):
        model_one = _TestModelOne(10, 20, 3)
        model_two = _TestModelTwo(10, 20, 10, 4)
        model_one.to(device_0)
        model_two.to(device_1)
        # test skip layer.bias
        model_dict, ch, unch = compatible_mod_state(model_one, model_two, mapping={"layer.bias": None})
        model_one.load_state_dict(model_dict)
        x = torch.randn((4, 10), device=device_0)
        output = model_one(x).detach().cpu().numpy()
        expected = np.array(
            [
                [0.40672818, -0.03851359, -0.5709822],
                [-0.6806116, -0.10745992, -0.70241666],
                [0.08895908, -0.38165253, -0.45776233],
                [0.26456985, -0.34086746, -0.02175514],
            ]
        )
        np.testing.assert_allclose(output, expected, atol=1e-3)
        self.assertEqual(len(ch), 1)
        self.assertEqual(len(unch), 3)

    @parameterized.expand(TEST_CASES)
    def test_set_map_across(self, device_0, device_1):
        model_one = _TestModelOne(10, 10, 3)
        model_two = _TestModelTwo(10, 10, 10, 4)
        model_one.to(device_0)
        model_two.to(device_1)
        # test weight map
        model_dict, ch, unch = compatible_mod_state(
            model_one, model_two, mapping={"layer_1.weight": "layer.weight", "layer_1.bias": "layer_1.weight"}
        )
        model_one.load_state_dict(model_dict)
        x = torch.randn((4, 10), device=device_0)
        output = model_one(x).detach().cpu().numpy()
        expected = np.array(
            [
                [0.05807909, 0.7679508, 0.30770922],
                [0.38950497, 0.5880758, 0.3073718],
                [0.26307103, 0.81471455, 0.1975978],
                [0.38134947, 0.26907563, 0.48251086],
            ]
        )
        np.testing.assert_allclose(output, expected, atol=1e-3)
        self.assertEqual(len(ch), 2)
        self.assertEqual(len(unch), 2)


if __name__ == "__main__":
    unittest.main()

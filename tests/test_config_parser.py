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

import torch
from parameterized import parameterized

from monai.apps import ConfigParser
from monai.transforms import LoadImaged

TEST_CASES = [
    # test MONAI components
    [
        dict(pkgs=["torch", "monai"], modules=["transforms"]),
        {"name": "LoadImaged", "args": {"keys": ["image"]}},
        LoadImaged,
    ],
    # test non-monai modules
    [
        dict(pkgs=["torch", "monai"], modules=["optim"]),
        {"name": "Adam", "args": {"params": torch.nn.PReLU().parameters(), "lr": 1e-4}},
        torch.optim.Adam,
    ],
    # test python `path`
    [dict(pkgs=[], modules=[]), {"path": "monai.transforms.LoadImaged", "args": {"keys": ["image"]}}, LoadImaged],
    # test `disabled`
    [
        dict(pkgs=["torch", "monai"], modules=["transforms"]),
        {"name": "LoadImaged", "disabled": True, "args": {"keys": ["image"]}},
        None,
    ],
]


class TestConfigParser(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_type(self, input_param, test_input, output_type):
        configer = ConfigParser(**input_param)
        result = configer.build_component(test_input)
        if result is not None:
            self.assertTrue(isinstance(result, output_type))
            if isinstance(result, LoadImaged):
                self.assertEqual(result.keys[0], "image")
        else:
            # test `disabled` works fine
            self.assertEqual(result, output_type)


if __name__ == "__main__":
    unittest.main()

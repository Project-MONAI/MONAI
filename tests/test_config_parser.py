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
from tests.utils import skip_if_windows

TEST_CASE_1 = [
    dict(pkgs=["monai"], modules=["transforms"]),
    {"name": "LoadImaged", "args": {"keys": ["image"]}},
    LoadImaged,
]
# test python `path`
TEST_CASE_2 = [
    dict(pkgs=[], modules=[]),
    {"path": "monai.transforms.LoadImaged", "args": {"keys": ["image"]}},
    LoadImaged,
]
# test `disabled`
TEST_CASE_3 = [
    dict(pkgs=["monai"], modules=["transforms"]),
    {"name": "LoadImaged", "disabled": True, "args": {"keys": ["image"]}},
    None,
]
# test non-monai modules
TEST_CASE_4 = [
    dict(pkgs=["torch.optim", "monai"], modules=["adam"]),
    {"name": "Adam", "args": {"params": torch.nn.PReLU().parameters(), "lr": 1e-4}},
    torch.optim.Adam,
]


class TestConfigParser(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_3])
    def test_type(self, input_param, test_input, output_type):
        configer = ConfigParser(**input_param)
        result = configer.build_component(test_input)
        if result is not None:
            self.assertTrue(isinstance(result, output_type))
            self.assertEqual(result.keys[0], "image")
        else:
            # test `disabled` works fine
            self.assertEqual(result, output_type)

    @skip_if_windows
    @parameterized.expand([TEST_CASE_4])
    def test_non_monai(self, input_param, test_input, output_type):
        configer = ConfigParser(**input_param)
        result = configer.build_component(test_input)
        self.assertTrue(isinstance(result, output_type))


if __name__ == "__main__":
    unittest.main()

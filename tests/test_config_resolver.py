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

from distutils.command.config import config
from typing import Callable, Iterator
import unittest

import torch
import monai
from parameterized import parameterized

from monai.apps import ConfigComponent, ConfigResolver, ModuleScanner
from monai.data import Dataset, DataLoader
from monai.transforms import LoadImaged

# test instance with no reference
TEST_CASE_1 = [
    {
        "transform#1": {"<name>": "LoadImaged", "<args>": {"keys": ["image"]}},
        "transform#1#<name>": "LoadImaged",
        "transform#1#<args>": {"keys": ["image"]},
        "transform#1#<args>#keys": ["image"],
        "transform#1#<args>#keys#0": "image",
    },
    "transform#1",
    LoadImaged,
]


class TestConfigComponent(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1])
    def test_get_instance(self, configs, expected_id, output_type):
        scanner = ModuleScanner(pkgs=["torch.optim", "monai"], modules=["data", "transforms", "adam"])
        resolver = ConfigResolver()
        for k, v in configs.items():
            resolver.update(ConfigComponent(
                id=k, config=v, module_scanner=scanner, globals={"monai": monai, "torch": torch}
            ))
        config, ins = resolver.resolve_one_component(expected_id)
        self.assertTrue(isinstance(ins, output_type))
        # test lazy instantiation
        config["<disabled>"] = False
        ins = ConfigComponent(id=expected_id, module_scanner=scanner, config=config).build()
        self.assertTrue(isinstance(ins, output_type))


if __name__ == "__main__":
    unittest.main()

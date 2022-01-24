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

import monai
from monai.apps import ConfigComponent, ConfigResolver, ModuleScanner
from monai.data import DataLoader
from monai.transforms import LoadImaged, RandTorchVisiond
from monai.utils import optional_import

_, has_tv = optional_import("torchvision")

# test instance with no dependencies
TEST_CASE_1 = [
    {
        # all the recursively parsed config items
        "transform#1": {"<name>": "LoadImaged", "<args>": {"keys": ["image"]}},
        "transform#1#<name>": "LoadImaged",
        "transform#1#<args>": {"keys": ["image"]},
        "transform#1#<args>#keys": ["image"],
        "transform#1#<args>#keys#0": "image",
    },
    "transform#1",
    LoadImaged,
]
# test depends on other component and executable code
TEST_CASE_2 = [
    {
        # all the recursively parsed config items
        "dataloader": {
            "<name>": "DataLoader",
            "<args>": {"dataset": "@dataset", "collate_fn": "$monai.data.list_data_collate"},
        },
        "dataset": {"<name>": "Dataset", "<args>": {"data": [1, 2]}},
        "dataloader#<name>": "DataLoader",
        "dataloader#<args>": {"dataset": "@dataset", "collate_fn": "$monai.data.list_data_collate"},
        "dataloader#<args>#dataset": "@dataset",
        "dataloader#<args>#collate_fn": "$monai.data.list_data_collate",
        "dataset#<name>": "Dataset",
        "dataset#<args>": {"data": [1, 2]},
        "dataset#<args>#data": [1, 2],
        "dataset#<args>#data#0": 1,
        "dataset#<args>#data#1": 2,
    },
    "dataloader",
    DataLoader,
]
# test config has key `name`
TEST_CASE_3 = [
    {
        # all the recursively parsed config items
        "transform#1": {
            "<name>": "RandTorchVisiond",
            "<args>": {"keys": "image", "name": "ColorJitter", "brightness": 0.25},
        },
        "transform#1#<name>": "RandTorchVisiond",
        "transform#1#<args>": {"keys": "image", "name": "ColorJitter", "brightness": 0.25},
        "transform#1#<args>#keys": "image",
        "transform#1#<args>#name": "ColorJitter",
        "transform#1#<args>#brightness": 0.25,
    },
    "transform#1",
    RandTorchVisiond,
]


class TestConfigComponent(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1, TEST_CASE_2] + ([TEST_CASE_3] if has_tv else []))
    def test_resolve(self, configs, expected_id, output_type):
        scanner = ModuleScanner(pkgs=["torch.optim", "monai"], modules=["data", "transforms", "adam"])
        resolver = ConfigResolver()
        for k, v in configs.items():
            resolver.add(
                ConfigComponent(id=k, config=v, module_scanner=scanner, globals={"monai": monai, "torch": torch})
            )
        ins = resolver.get_resolved_component(expected_id)
        self.assertTrue(isinstance(ins, output_type))
        # test resolve all
        resolver.resolved_configs = {}
        resolver.resolved_components = {}
        resolver.resolve_all()
        ins = resolver.get_resolved_component(expected_id)
        self.assertTrue(isinstance(ins, output_type))
        # test lazy instantiation
        config = resolver.get_resolved_config(expected_id)
        config["<disabled>"] = False
        ins = ConfigComponent(id=expected_id, module_scanner=scanner, config=config).build()
        self.assertTrue(isinstance(ins, output_type))

    def test_circular_dependencies(self):
        scanner = ModuleScanner(pkgs=[], modules=[])
        resolver = ConfigResolver()
        configs = {"A": "@B", "B": "@C", "C": "@A"}
        for k, v in configs.items():
            resolver.add(ConfigComponent(id=k, config=v, module_scanner=scanner))
        for k in ["A", "B", "C"]:
            with self.assertRaises(ValueError):
                resolver.get_resolved_component(k)


if __name__ == "__main__":
    unittest.main()

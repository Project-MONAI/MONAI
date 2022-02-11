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
from functools import partial
from typing import Callable, Iterator

import torch
from parameterized import parameterized
from torch.optim._multi_tensor import Adam

import monai
from monai.apps import ConfigComponent
from monai.data import DataLoader, Dataset
from monai.transforms import LoadImaged, RandTorchVisiond
from monai.utils import ComponentScanner, optional_import

_, has_tv = optional_import("torchvision")

TEST_CASE_1 = [
    dict(pkgs=["monai"], modules=["transforms"]),
    {"<name>": "LoadImaged", "<args>": {"keys": ["image"]}},
    LoadImaged,
]
# test python `<path>`
TEST_CASE_2 = [
    dict(pkgs=[], modules=[]),
    {"<path>": "monai.transforms.LoadImaged", "<args>": {"keys": ["image"]}},
    LoadImaged,
]
# test `<disabled>`
TEST_CASE_3 = [
    dict(pkgs=["monai"], modules=["transforms"]),
    {"<name>": "LoadImaged", "<disabled>": True, "<args>": {"keys": ["image"]}},
    dict,
]
# test unresolved dependency
TEST_CASE_4 = [
    dict(pkgs=["monai"], modules=["transforms"]),
    {"<name>": "LoadImaged", "<args>": {"keys": ["@key_name"]}},
    dict,
]
# test non-monai modules
TEST_CASE_5 = [
    dict(pkgs=["torch.optim", "monai"], modules=["adam"]),
    {"<name>": "Adam", "<args>": {"params": torch.nn.PReLU().parameters(), "lr": 1e-4}},
    Adam,
]
TEST_CASE_6 = [
    dict(pkgs=["monai"], modules=["data"]),
    {"<name>": "decollate_batch", "<args>": {"detach": True, "pad": True}},
    partial,
]
# test args contains "name" field
TEST_CASE_7 = [
    dict(pkgs=["monai"], modules=["transforms"]),
    {"<name>": "RandTorchVisiond", "<args>": {"keys": "image", "name": "ColorJitter", "brightness": 0.25}},
    RandTorchVisiond,
]
# test dependencies of dict config
TEST_CASE_8 = [{"dataset": "@dataset", "batch_size": 2}, ["test#dataset", "dataset", "test#batch_size"]]
# test dependencies of list config
TEST_CASE_9 = [
    {"dataset": "@dataset", "transforms": ["@trans0", "@trans1"]},
    ["test#dataset", "dataset", "test#transforms", "test#transforms#0", "trans0", "test#transforms#1", "trans1"],
]
# test dependencies of execute code
TEST_CASE_10 = [
    {"dataset": "$@dataset.test_func()", "transforms": ["$torch.zeros([2, 2]) + @trans"]},
    ["test#dataset", "dataset", "test#transforms", "test#transforms#0", "trans"],
]

# test dependencies of lambda function
TEST_CASE_11 = [
    {"lr_range": "$lambda x: x + @num_epochs", "lr": ["$lambda x: torch.zeros([2, 2]) + @init_lr"]},
    ["test#lr_range", "num_epochs", "test#lr", "test#lr#0", "init_lr"],
]
# test instance with no dependencies
TEST_CASE_12 = [
    "transform#1",
    {"<name>": "LoadImaged", "<args>": {"keys": ["image"]}},
    {"transform#1#<name>": "LoadImaged", "transform#1#<args>": {"keys": ["image"]}},
    LoadImaged,
]
# test dataloader refers to `@dataset`, here we don't test recursive dependencies, test that in `ConfigResolver`
TEST_CASE_13 = [
    "dataloader",
    {"<name>": "DataLoader", "<args>": {"dataset": "@dataset", "batch_size": 2}},
    {"dataloader#<name>": "DataLoader", "dataloader#<args>": {"dataset": Dataset(data=[1, 2]), "batch_size": 2}},
    DataLoader,
]
# test dependencies in code execution
TEST_CASE_14 = [
    "optimizer",
    {"<name>": "Adam", "<args>": {"params": "$@model.parameters()", "lr": "@learning_rate"}},
    {"optimizer#<name>": "Adam", "optimizer#<args>": {"params": torch.nn.PReLU().parameters(), "lr": 1e-4}},
    Adam,
]
# test replace dependencies with code execution result
TEST_CASE_15 = ["optimizer#<args>#params", "$@model.parameters()", {"model": torch.nn.PReLU()}, Iterator]
# test execute some function in args, test pre-imported global packages `monai`
TEST_CASE_16 = ["dataloader#<args>#collate_fn", "$monai.data.list_data_collate", {}, Callable]
# test lambda function, should not execute the lambda function, just change the string with dependent objects
TEST_CASE_17 = ["dataloader#<args>#collate_fn", "$lambda x: monai.data.list_data_collate(x) + 100", {}, Callable]


class TestConfigComponent(unittest.TestCase):
    @parameterized.expand(
        [TEST_CASE_1, TEST_CASE_2, TEST_CASE_3, TEST_CASE_4, TEST_CASE_5, TEST_CASE_6]
        + ([TEST_CASE_7] if has_tv else [])
    )
    def test_build(self, input_param, test_input, output_type):
        scanner = ComponentScanner(**input_param)
        configer = ConfigComponent(id="test", config=test_input, scanner=scanner)
        ret = configer.build()
        self.assertTrue(isinstance(ret, output_type))
        if isinstance(ret, LoadImaged):
            self.assertEqual(ret.keys[0], "image")
        if isinstance(ret, dict):
            # test `<disabled>` works fine
            self.assertDictEqual(ret, test_input)

    @parameterized.expand([TEST_CASE_8, TEST_CASE_9, TEST_CASE_10, TEST_CASE_11])
    def test_dependent_ids(self, test_input, ref_ids):
        scanner = ComponentScanner(pkgs=[], modules=[])
        configer = ConfigComponent(id="test", config=test_input, scanner=scanner)
        ret = configer.get_dependent_ids()
        self.assertListEqual(ret, ref_ids)

    @parameterized.expand([TEST_CASE_12, TEST_CASE_13, TEST_CASE_14, TEST_CASE_15, TEST_CASE_16, TEST_CASE_17])
    def test_update_dependencies(self, id, test_input, deps, output_type):
        scanner = ComponentScanner(pkgs=["torch.optim", "monai"], modules=["data", "transforms", "adam"])
        configer = ConfigComponent(id=id, config=test_input, scanner=scanner, globals={"monai": monai, "torch": torch})
        config = configer.get_updated_config(deps)
        ret = configer.build(config)
        self.assertTrue(isinstance(ret, output_type))


if __name__ == "__main__":
    unittest.main()

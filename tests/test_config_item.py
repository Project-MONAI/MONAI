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

import monai
from monai.apps import ComponentLocator, ConfigComponent, ConfigItem, is_instantiable
from monai.data import DataLoader, Dataset
from monai.transforms import LoadImaged, RandTorchVisiond
from monai.utils import optional_import

_, has_tv = optional_import("torchvision")

TEST_CASE_1 = [{"<name>": "LoadImaged", "<args>": {"keys": ["image"]}}, LoadImaged]
# test python `<path>`
TEST_CASE_2 = [{"<path>": "monai.transforms.LoadImaged", "<args>": {"keys": ["image"]}}, LoadImaged]
# test `<disabled>`
TEST_CASE_3 = [{"<name>": "LoadImaged", "<disabled>": True, "<args>": {"keys": ["image"]}}, dict]
# test unresolved reference
TEST_CASE_4 = [{"<name>": "LoadImaged", "<args>": {"keys": ["@key_name"]}}]
# test non-monai modules and excludes
TEST_CASE_5 = [
    {"<path>": "torch.optim.Adam", "<args>": {"params": torch.nn.PReLU().parameters(), "lr": 1e-4}},
    torch.optim.Adam,
]
TEST_CASE_6 = [{"<name>": "decollate_batch", "<args>": {"detach": True, "pad": True}}, partial]
# test args contains "name" field
TEST_CASE_7 = [
    {"<name>": "RandTorchVisiond", "<args>": {"keys": "image", "name": "ColorJitter", "brightness": 0.25}},
    RandTorchVisiond,
]
# test references of dict config
TEST_CASE_8 = [{"dataset": "@dataset", "batch_size": 2}, ["dataset"]]
# test references of list config
TEST_CASE_9 = [{"dataset": "@dataset", "transforms": ["@trans0", "@trans1"]}, ["dataset", "trans0", "trans1"]]
# test references of execute code
TEST_CASE_10 = [
    {"dataset": "$@dataset.test_func()", "transforms": ["$torch.zeros([2, 2]) + @trans"]},
    ["dataset", "trans"],
]
# test references of lambda function
TEST_CASE_11 = [
    {"lr_range": "$lambda x: x + @num_epochs", "lr": ["$lambda x: torch.zeros([2, 2]) + @init_lr"]},
    ["num_epochs", "init_lr"],
]
# test instance with no references
TEST_CASE_12 = ["transform#1", {"<name>": "LoadImaged", "<args>": {"keys": ["image"]}}, {}, LoadImaged]
# test dataloader refers to `@dataset`, here we don't test recursive references, test that in `ConfigResolver`
TEST_CASE_13 = [
    "dataloader",
    {"<name>": "DataLoader", "<args>": {"dataset": "@dataset", "batch_size": 2}},
    {"dataset": Dataset(data=[1, 2])},
    DataLoader,
]
# test references in code execution
TEST_CASE_14 = [
    "optimizer",
    {"<path>": "torch.optim.Adam", "<args>": {"params": "$@model.parameters()", "lr": "@learning_rate"}},
    {"model": torch.nn.PReLU(), "learning_rate": 1e-4},
    torch.optim.Adam,
]
# test replace references with code execution result
TEST_CASE_15 = ["optimizer#<args>#params", "$@model.parameters()", {"model": torch.nn.PReLU()}, Iterator]
# test execute some function in args, test pre-imported global packages `monai`
TEST_CASE_16 = ["dataloader#<args>#collate_fn", "$monai.data.list_data_collate", {}, Callable]
# test lambda function, should not execute the lambda function, just change the string with referring objects
TEST_CASE_17 = ["dataloader#<args>#collate_fn", "$lambda x: monai.data.list_data_collate(x) + 100", {}, Callable]


class TestConfigItem(unittest.TestCase):
    @parameterized.expand(
        [TEST_CASE_1, TEST_CASE_2, TEST_CASE_3, TEST_CASE_5, TEST_CASE_6] + ([TEST_CASE_7] if has_tv else [])
    )
    def test_instantiate(self, test_input, output_type):
        locator = ComponentLocator(excludes=["metrics"])
        configer = ConfigComponent(id="test", config=test_input, locator=locator)
        ret = configer.instantiate()
        if test_input.get("<disabled>", False):
            # test `<disabled>` works fine
            self.assertEqual(ret, None)
            return
        self.assertTrue(isinstance(ret, output_type))
        if isinstance(ret, LoadImaged):
            self.assertEqual(ret.keys[0], "image")

    @parameterized.expand([TEST_CASE_4])
    def test_raise_error(self, test_input):
        with self.assertRaises(KeyError):  # has unresolved keys
            configer = ConfigItem(id="test", config=test_input)
            configer.resolve()

    @parameterized.expand([TEST_CASE_8, TEST_CASE_9, TEST_CASE_10, TEST_CASE_11])
    def test_referring_ids(self, test_input, ref_ids):
        configer = ConfigItem(id="test", config=test_input)  # also test default locator
        ret = configer.get_id_of_refs()
        self.assertListEqual(ret, ref_ids)

    @parameterized.expand([TEST_CASE_12, TEST_CASE_13, TEST_CASE_14, TEST_CASE_15, TEST_CASE_16, TEST_CASE_17])
    def test_resolve_references(self, id, test_input, refs, output_type):
        configer = ConfigComponent(
            id=id, config=test_input, locator=None, excludes=["utils"], globals={"monai": monai, "torch": torch}
        )
        configer.resolve(refs=refs)
        ret = configer.get_resolved_config()
        if is_instantiable(ret):
            ret = configer.instantiate(**{})  # also test kwargs
        self.assertTrue(isinstance(ret, output_type))

    def test_lazy_instantiation(self):
        config = {"<name>": "DataLoader", "<args>": {"dataset": "@dataset", "batch_size": 2}}
        refs = {"dataset": Dataset(data=[1, 2])}
        configer = ConfigComponent(config=config, locator=None)
        init_config = configer.get_config()
        # modify config content at runtime
        init_config["<args>"]["batch_size"] = 4
        configer.update_config(config=init_config)

        configer.resolve(refs=refs)
        ret = configer.instantiate()
        self.assertTrue(isinstance(ret, DataLoader))
        self.assertEqual(ret.batch_size, 4)


if __name__ == "__main__":
    unittest.main()

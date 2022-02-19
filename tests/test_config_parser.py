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

import unittest
from unittest import skipUnless

from parameterized import parameterized

from monai.apps import ConfigParser
from monai.apps.manifest.config_item import ConfigComponent
from monai.data import DataLoader, Dataset
from monai.transforms import Compose, LoadImaged, RandTorchVisiond
from monai.utils import optional_import

_, has_tv = optional_import("torchvision")

# test the resolved and parsed instances
TEST_CASE_1 = [
    {
        "transform": {
            "<name>": "Compose",
            "<args>": {
                "transforms": [
                    {"<name>": "LoadImaged", "<args>": {"keys": "image"}},
                    {
                        "<name>": "RandTorchVisiond",
                        "<args>": {"keys": "image", "name": "ColorJitter", "brightness": 0.25},
                    },
                ]
            },
        },
        "dataset": {"<name>": "Dataset", "<args>": {"data": [1, 2], "transform": "@transform"}},
        "dataloader": {
            "<name>": "DataLoader",
            "<args>": {"dataset": "@dataset", "batch_size": 2, "collate_fn": "monai.data.list_data_collate"},
        },
    },
    ["transform", "transform#<args>#transforms#0", "transform#<args>#transforms#1", "dataset", "dataloader"],
    [Compose, LoadImaged, RandTorchVisiond, Dataset, DataLoader],
]


class TestConfigComponent(unittest.TestCase):
    def test_config_content(self):
        parser = ConfigParser(config={})
        test_config = {"preprocessing": [{"<name>": "LoadImage"}], "dataset": {"<name>": "Dataset"}}
        parser.update_config(config=test_config)
        self.assertEqual(str(parser.get_config()), str(test_config))
        parser.update_config(config={"<name>": "CacheDataset"}, id="dataset")
        self.assertDictEqual(parser.get_config(id="dataset"), {"<name>": "CacheDataset"})
        parser.update_config(config="Dataset", id="dataset#<name>")
        self.assertEqual(parser.get_config(id="dataset#<name>"), "Dataset")

    @parameterized.expand([TEST_CASE_1])
    @skipUnless(has_tv, "Requires torchvision.")
    def test_parse(self, config, expected_ids, output_types):
        parser = ConfigParser(config=config, globals={"monai": "monai"})
        for id, cls in zip(expected_ids, output_types):
            item = parser.get_config_item(id, resolve=True)
            # test lazy instantiation
            if isinstance(item, ConfigComponent):
                self.assertTrue(isinstance(item.instantiate(), cls))
            # test get instance directly
            self.assertTrue(isinstance(parser.get_resolved_content(id), cls))


if __name__ == "__main__":
    unittest.main()

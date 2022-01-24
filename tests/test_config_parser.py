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
from parameterized import parameterized

from monai.apps import ConfigParser
from monai.data import Dataset, DataLoader
from monai.transforms import Compose, LoadImaged, RandTorchVisiond

# test the resolved and parsed instances
TEST_CASE_1 = [
    {
        "transform": {
            "<name>": "Compose",
            "<args>": {"transforms": [
                {"<name>": "LoadImaged", "<args>": {"keys": "image"}},
                {"<name>": "RandTorchVisiond", "<args>": {"keys": "image", "name": "ColorJitter", "brightness": 0.25}},
            ]}
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
        parser = ConfigParser(pkgs=["torch.optim", "monai"], modules=["data", "transforms", "adam"])
        test_config = {"preprocessing": [{"name": "LoadImage"}], "dataset": {"name": "Dataset"}}
        parser.set_config(config=test_config)
        self.assertEqual(str(parser.get_config()), str(test_config))
        parser.set_config(config={"name": "CacheDataset"}, id="preprocessing#0#datasets")
        self.assertDictEqual(parser.get_config(id="preprocessing#0#datasets"), {"name": "CacheDataset"})

    @parameterized.expand([TEST_CASE_1])
    def test_parse(self, config, expected_ids, output_types):
        parser = ConfigParser(
            pkgs=["torch.optim", "monai"], modules=["data", "transforms", "adam"], global_imports=["monai"], config=config
        )
        for id, cls in zip(expected_ids, output_types):
            config = parser.get_resolved_config(id)
            # test lazy instantiation
            self.assertTrue(isinstance(config, dict))
            self.assertTrue(isinstance(parser.build(config), cls))
            # test get instance directly
            self.assertTrue(isinstance(parser.get_resolved_component(id), cls))


if __name__ == "__main__":
    unittest.main()

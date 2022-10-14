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

import json
import os
import unittest

from parameterized import parameterized

from monai.fl.client import MonaiAlgoStats
from monai.fl.utils.constants import ExtraItems, FlStatistics
from monai.fl.utils.exchange_object import ExchangeObject
from tests.utils import SkipIfNoModule

TEST_GET_DATA_STATS_1 = [
    {
        "bundle_root": os.path.join(os.path.dirname(__file__)),
        "config_train_filename": os.path.join("testing_data", "config_fl_stats_1.json"),
        "config_filters_filename": os.path.join("testing_data", "config_fl_filters.json"),
    }
]
TEST_GET_DATA_STATS_2 = [
    {
        "bundle_root": os.path.join(os.path.dirname(__file__)),
        "config_train_filename": os.path.join("testing_data", "config_fl_stats_2.json"),
        "config_filters_filename": os.path.join("testing_data", "config_fl_filters.json"),
    }
]
TEST_GET_DATA_STATS_3 = [
    {
        "bundle_root": os.path.join(os.path.dirname(__file__)),
        "config_train_filename": [
            os.path.join("testing_data", "config_fl_stats_1.json"),
            os.path.join("testing_data", "config_fl_stats_2.json"),
        ],
        "config_filters_filename": [
            os.path.join("testing_data", "config_fl_filters.json"),
            os.path.join("testing_data", "config_fl_filters.json"),
        ],
    }
]


@SkipIfNoModule("ignite")
class TestFLMonaiAlgo(unittest.TestCase):
    @parameterized.expand([TEST_GET_DATA_STATS_1, TEST_GET_DATA_STATS_2, TEST_GET_DATA_STATS_3])
    def test_get_data_stats(self, input_params):
        # get testing data dir and update train config; using the first to define data dir
        if input_params["config_train_filename"]:
            if isinstance(input_params["config_train_filename"], list):
                config_train_filename = input_params["config_train_filename"][0]
            else:
                config_train_filename = input_params["config_train_filename"]
            with open(os.path.join(input_params["bundle_root"], config_train_filename)) as f:
                config_train = json.load(f)

            config_train["dataset_dir"] = os.path.join(os.path.dirname(__file__), "testing_data")

            with open(os.path.join(input_params["bundle_root"], config_train_filename), "w") as f:
                json.dump(config_train, f, indent=4)

        # initialize algo
        algo = MonaiAlgoStats(**input_params)
        algo.initialize(extra={ExtraItems.CLIENT_NAME: "test_fl"})

        requested_stats = {FlStatistics.HIST_BINS: 100, FlStatistics.HIST_RANGE: [-500, 500]}
        # test train
        stats = algo.get_data_stats(extra=requested_stats)
        self.assertIsInstance(stats, ExchangeObject)


if __name__ == "__main__":
    unittest.main()

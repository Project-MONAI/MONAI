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

from monai.bundle import ConfigParser
from monai.fl.client.monai_algo import MonaiAlgo
from monai.fl.utils.constants import ExtraItems
from monai.fl.utils.exchange_object import ExchangeObject
from tests.utils import SkipIfNoModule

TEST_TRAIN_1 = [
    {
        "config_train_file": os.path.join(os.path.dirname(__file__), "testing_data", "config_fl_train.json"),
        "config_filters_file": os.path.join(os.path.dirname(__file__), "testing_data", "config_fl_filters.json"),
    }
]
TEST_TRAIN_2 = [
    {
        "config_train_file": os.path.join(os.path.dirname(__file__), "testing_data", "config_fl_train.json"),
        "config_filters_file": None,
    }
]

TEST_EVALUATE_1 = [
    {
        "config_evaluate_file": os.path.join(os.path.dirname(__file__), "testing_data", "config_fl_evaluate.json"),
        "config_filters_file": os.path.join(os.path.dirname(__file__), "testing_data", "config_fl_filters.json"),
    }
]
TEST_EVALUATE_2 = [
    {
        "config_evaluate_file": os.path.join(os.path.dirname(__file__), "testing_data", "config_fl_evaluate.json"),
        "config_filters_file": None,
    }
]

TEST_GET_WEIGHTS_1 = [
    {
        "config_train_file": os.path.join(os.path.dirname(__file__), "testing_data", "config_fl_train.json"),
        "config_filters_file": os.path.join(os.path.dirname(__file__), "testing_data", "config_fl_filters.json"),
    }
]
TEST_GET_WEIGHTS_2 = [
    {
        "config_train_file": None,
        "config_filters_file": os.path.join(os.path.dirname(__file__), "testing_data", "config_fl_filters.json"),
    }
]


@SkipIfNoModule("ignite")
class TestFLMonaiAlgo(unittest.TestCase):
    @parameterized.expand([TEST_TRAIN_1, TEST_TRAIN_2])
    def test_train(self, input_params):
        # get testing data dir and update train config
        with open(input_params["config_train_file"]) as f:
            config_train = json.load(f)

        config_train["dataset_dir"] = os.path.join(os.path.dirname(__file__), "testing_data")

        with open(input_params["config_train_file"], "w") as f:
            json.dump(config_train, f, indent=4)

        # initialize algo
        algo = MonaiAlgo(**input_params)
        algo.initialize(extra={ExtraItems.CLIENT_NAME: "test_fl"})

        # initialize model
        parser = ConfigParser()
        parser.read_config(input_params["config_train_file"])
        parser.parse()
        network = parser.get_parsed_content("network")

        data = ExchangeObject(weights=network.state_dict())

        # test train
        algo.train(data=data, extra={})

    @parameterized.expand([TEST_EVALUATE_1, TEST_EVALUATE_2])
    def test_evaluate(self, input_params):
        # get testing data dir and update train config
        with open(input_params["config_evaluate_file"]) as f:
            config_evaluate = json.load(f)

        config_evaluate["dataset_dir"] = os.path.join(os.path.dirname(__file__), "testing_data")

        with open(input_params["config_evaluate_file"], "w") as f:
            json.dump(config_evaluate, f, indent=4)

        # initialize algo
        algo = MonaiAlgo(**input_params)
        algo.initialize(extra={ExtraItems.CLIENT_NAME: "test_fl"})

        # initialize model
        parser = ConfigParser()
        parser.read_config(input_params["config_evaluate_file"])
        parser.parse()
        network = parser.get_parsed_content("network")

        data = ExchangeObject(weights=network.state_dict())

        # test evaluate
        algo.evaluate(data=data, extra={})

    @parameterized.expand([TEST_GET_WEIGHTS_1, TEST_GET_WEIGHTS_2])
    def test_get_weights(self, input_params):
        # get testing data dir and update train config
        if input_params["config_train_file"]:
            with open(input_params["config_train_file"]) as f:
                config_train = json.load(f)

            config_train["dataset_dir"] = os.path.join(os.path.dirname(__file__), "testing_data")

            with open(input_params["config_train_file"], "w") as f:
                json.dump(config_train, f, indent=4)

        # initialize algo
        algo = MonaiAlgo(**input_params)
        algo.initialize(extra={ExtraItems.CLIENT_NAME: "test_fl"})

        # test train
        weights = algo.get_weights(extra={})
        self.assertIsInstance(weights, ExchangeObject)

    # TODO: test abort and finalize


if __name__ == "__main__":
    unittest.main()

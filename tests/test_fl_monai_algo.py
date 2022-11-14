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

import os
import unittest

from parameterized import parameterized

from monai.bundle import ConfigParser
from monai.fl.client.monai_algo import MonaiAlgo
from monai.fl.utils.constants import ExtraItems
from monai.fl.utils.exchange_object import ExchangeObject
from tests.utils import SkipIfNoModule

_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
_data_dir = os.path.join(_root_dir, "testing_data")

TEST_TRAIN_1 = [
    {
        "bundle_root": _data_dir,
        "config_train_filename": os.path.join(_data_dir, "config_fl_train.json"),
        "config_evaluate_filename": None,
        "config_filters_filename": os.path.join(_data_dir, "config_fl_filters.json"),
    }
]
TEST_TRAIN_2 = [
    {
        "bundle_root": _data_dir,
        "config_train_filename": os.path.join(_data_dir, "config_fl_train.json"),
        "config_evaluate_filename": None,
        "config_filters_filename": None,
    }
]
TEST_TRAIN_3 = [
    {
        "bundle_root": _data_dir,
        "config_train_filename": [
            os.path.join(_data_dir, "config_fl_train.json"),
            os.path.join(_data_dir, "config_fl_train.json"),
        ],
        "config_evaluate_filename": None,
        "config_filters_filename": [
            os.path.join(_data_dir, "config_fl_filters.json"),
            os.path.join(_data_dir, "config_fl_filters.json"),
        ],
    }
]

TEST_EVALUATE_1 = [
    {
        "bundle_root": _data_dir,
        "config_train_filename": None,
        "config_evaluate_filename": os.path.join(_data_dir, "config_fl_evaluate.json"),
        "config_filters_filename": os.path.join(_data_dir, "config_fl_filters.json"),
    }
]
TEST_EVALUATE_2 = [
    {
        "bundle_root": _data_dir,
        "config_train_filename": None,
        "config_evaluate_filename": os.path.join(_data_dir, "config_fl_evaluate.json"),
        "config_filters_filename": None,
    }
]
TEST_EVALUATE_3 = [
    {
        "bundle_root": _data_dir,
        "config_train_filename": None,
        "config_evaluate_filename": [
            os.path.join(_data_dir, "config_fl_evaluate.json"),
            os.path.join(_data_dir, "config_fl_evaluate.json"),
        ],
        "config_filters_filename": [
            os.path.join(_data_dir, "config_fl_filters.json"),
            os.path.join(_data_dir, "config_fl_filters.json"),
        ],
    }
]

TEST_GET_WEIGHTS_1 = [
    {
        "bundle_root": _data_dir,
        "config_train_filename": os.path.join(_data_dir, "config_fl_train.json"),
        "config_evaluate_filename": None,
        "send_weight_diff": False,
        "config_filters_filename": os.path.join(_data_dir, "config_fl_filters.json"),
    }
]
TEST_GET_WEIGHTS_2 = [
    {
        "bundle_root": _data_dir,
        "config_train_filename": None,
        "config_evaluate_filename": None,
        "send_weight_diff": False,
        "config_filters_filename": os.path.join(_data_dir, "config_fl_filters.json"),
    }
]
TEST_GET_WEIGHTS_3 = [
    {
        "bundle_root": _data_dir,
        "config_train_filename": os.path.join(_data_dir, "config_fl_train.json"),
        "config_evaluate_filename": None,
        "send_weight_diff": True,
        "config_filters_filename": os.path.join(_data_dir, "config_fl_filters.json"),
    }
]
TEST_GET_WEIGHTS_4 = [
    {
        "bundle_root": _data_dir,
        "config_train_filename": [
            os.path.join(_data_dir, "config_fl_train.json"),
            os.path.join(_data_dir, "config_fl_train.json"),
        ],
        "config_evaluate_filename": None,
        "send_weight_diff": True,
        "config_filters_filename": [
            os.path.join(_data_dir, "config_fl_filters.json"),
            os.path.join(_data_dir, "config_fl_filters.json"),
        ],
    }
]


@SkipIfNoModule("ignite")
class TestFLMonaiAlgo(unittest.TestCase):
    @parameterized.expand([TEST_TRAIN_1, TEST_TRAIN_2, TEST_TRAIN_3])
    def test_train(self, input_params):
        # get testing data dir and update train config; using the first to define data dir
        if isinstance(input_params["config_train_filename"], list):
            config_train_filename = [
                os.path.join(input_params["bundle_root"], x) for x in input_params["config_train_filename"]
            ]
        else:
            config_train_filename = os.path.join(input_params["bundle_root"], input_params["config_train_filename"])

        # initialize algo
        algo = MonaiAlgo(**input_params)
        algo.initialize(extra={ExtraItems.CLIENT_NAME: "test_fl"})
        algo.abort()

        # initialize model
        parser = ConfigParser()
        parser.read_config(config_train_filename)
        parser.parse()
        network = parser.get_parsed_content("network")

        data = ExchangeObject(weights=network.state_dict())

        # test train
        algo.train(data=data, extra={})
        algo.finalize()

    @parameterized.expand([TEST_EVALUATE_1, TEST_EVALUATE_2, TEST_EVALUATE_3])
    def test_evaluate(self, input_params):
        # get testing data dir and update train config; using the first to define data dir
        if isinstance(input_params["config_evaluate_filename"], list):
            config_eval_filename = [
                os.path.join(input_params["bundle_root"], x) for x in input_params["config_evaluate_filename"]
            ]
        else:
            config_eval_filename = os.path.join(input_params["bundle_root"], input_params["config_evaluate_filename"])

        # initialize algo
        algo = MonaiAlgo(**input_params)
        algo.initialize(extra={ExtraItems.CLIENT_NAME: "test_fl"})

        # initialize model
        parser = ConfigParser()
        parser.read_config(config_eval_filename)
        parser.parse()
        network = parser.get_parsed_content("network")

        data = ExchangeObject(weights=network.state_dict())

        # test evaluate
        algo.evaluate(data=data, extra={})

    @parameterized.expand([TEST_GET_WEIGHTS_1, TEST_GET_WEIGHTS_2, TEST_GET_WEIGHTS_3, TEST_GET_WEIGHTS_4])
    def test_get_weights(self, input_params):
        # initialize algo
        algo = MonaiAlgo(**input_params)
        algo.initialize(extra={ExtraItems.CLIENT_NAME: "test_fl"})

        # test train
        if input_params["send_weight_diff"]:  # should not work as test doesn't receive a global model
            with self.assertRaises(ValueError):
                weights = algo.get_weights(extra={})
        else:
            weights = algo.get_weights(extra={})
            self.assertIsInstance(weights, ExchangeObject)


if __name__ == "__main__":
    unittest.main()

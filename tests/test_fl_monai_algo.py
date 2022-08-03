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
from typing import Any, Dict

import torch
from ignite.engine import Engine
from parameterized import parameterized

from monai.bundle import ConfigParser
from monai.fl.client.monai_algo import MonaiAlgo
from monai.fl.utils.exchange_object import ExchangeObject
from monai.fl.utils.constants import ExtraItems

TEST_CASE_1 = [{"config_train_file": os.path.join(os.path.dirname(__file__), "testing_data", "config_fl_train.json"),
                "config_filters_file": os.path.join(os.path.dirname(__file__), "testing_data", "config_fl_filters.json")}]

class TestFLMonaiAlgo(unittest.TestCase):

    @parameterized.expand([TEST_CASE_1])
    def test_compute(self, input_params):
        algo = MonaiAlgo(**input_params)
        algo.initialize(extra={ExtraItems.CLIENT_NAME: "test_fl"})

        # initialize model
        parser = ConfigParser()
        parser.read_config(input_params["config_train_file"])
        parser.parse()
        network = parser.get_parsed_content("network")

        data = ExchangeObject(weights=network.state_dict())

        # test training
        algo.train(data=data, extra = {})

if __name__ == "__main__":
    unittest.main()

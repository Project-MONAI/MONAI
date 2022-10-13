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

import torch.distributed as dist
from parameterized import parameterized

from monai.bundle import ConfigParser
from monai.fl.client.monai_algo import MonaiAlgo
from monai.fl.utils.constants import ExtraItems
from monai.fl.utils.exchange_object import ExchangeObject
from tests.utils import DistCall, DistTestCase, SkipIfNoModule

TEST_TRAIN_1 = [
    {
        "bundle_root": os.path.join(os.path.dirname(__file__)),
        "config_train_filename": os.path.join("testing_data", "config_fl_train.json"),
        "config_evaluate_filename": None,
        "config_filters_filename": os.path.join("testing_data", "config_fl_filters.json"),
        "multi_gpu": True,
    }
]


@SkipIfNoModule("ignite")
class TestFLMonaiAlgo(DistTestCase):
    def setUp(self):
        input_params = TEST_TRAIN_1[0]
        # get testing data dir and update train config; using the first to define data dir
        if isinstance(input_params["config_train_filename"], list):
            config_train_filename = input_params["config_train_filename"][0]
        else:
            config_train_filename = input_params["config_train_filename"]
        with open(os.path.join(input_params["bundle_root"], config_train_filename)) as f:
            config_train = json.load(f)

        config_train["dataset_dir"] = os.path.join(os.path.dirname(__file__), "testing_data")

        with open(os.path.join(input_params["bundle_root"], config_train_filename), "w") as f:
            json.dump(config_train, f, indent=4)

    @parameterized.expand([TEST_TRAIN_1])
    @DistCall(nnodes=1, nproc_per_node=2, init_method="no_init")
    def test_train(self, input_params):
        # initialize algo
        algo = MonaiAlgo(**input_params)
        algo.initialize(extra={ExtraItems.CLIENT_NAME: "test_fl"})
        self.assertTrue(dist.get_rank() in (0, 1))

        # initialize model
        parser = ConfigParser()
        parser.read_config(os.path.join(input_params["bundle_root"], input_params["config_train_filename"]))
        parser.parse()
        network = parser.get_parsed_content("network")
        data = ExchangeObject(weights=network.state_dict())
        # test train
        algo.train(data=data, extra={})


if __name__ == "__main__":
    unittest.main()

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
from os.path import join as pathjoin

import torch.distributed as dist
from parameterized import parameterized

from monai.bundle import ConfigParser
from monai.fl.client.monai_algo import MonaiAlgo
from monai.fl.utils.constants import ExtraItems
from monai.fl.utils.exchange_object import ExchangeObject
from tests.utils import DistCall, DistTestCase, SkipIfNoModule, skip_if_no_cuda

_root_dir = pathjoin(os.path.dirname(__file__))
_data_dir = pathjoin(_root_dir, "testing_data")
TEST_TRAIN_1 = [
    {
        "bundle_root": _data_dir,
        "config_train_filename": [
            pathjoin(_data_dir, "config_fl_train.json"),
            pathjoin(_data_dir, "multi_gpu_train.json"),
        ],
        "config_evaluate_filename": None,
        "config_filters_filename": pathjoin(_root_dir, "testing_data", "config_fl_filters.json"),
        "multi_gpu": True,
    }
]

TEST_EVALUATE_1 = [
    {
        "bundle_root": _data_dir,
        "config_train_filename": None,
        "config_evaluate_filename": [
            pathjoin(_data_dir, "config_fl_evaluate.json"),
            pathjoin(_data_dir, "multi_gpu_evaluate.json"),
        ],
        "config_filters_filename": pathjoin(_data_dir, "config_fl_filters.json"),
        "multi_gpu": True,
    }
]


@SkipIfNoModule("ignite")
class TestFLMonaiAlgo(DistTestCase):
    @parameterized.expand([TEST_TRAIN_1])
    @DistCall(nnodes=1, nproc_per_node=2, init_method="no_init")
    @skip_if_no_cuda
    def test_train(self, input_params):
        # initialize algo
        algo = MonaiAlgo(**input_params)
        algo.initialize(extra={ExtraItems.CLIENT_NAME: "test_fl"})
        self.assertTrue(dist.get_rank() in (0, 1))

        # initialize model
        parser = ConfigParser()
        parser.read_config([pathjoin(input_params["bundle_root"], x) for x in input_params["config_train_filename"]])
        parser.parse()
        network = parser.get_parsed_content("network")
        data = ExchangeObject(weights=network.state_dict())
        # test train
        algo.train(data=data, extra={})

    @parameterized.expand([TEST_EVALUATE_1])
    @DistCall(nnodes=1, nproc_per_node=2, init_method="no_init")
    @skip_if_no_cuda
    def test_evaluate(self, input_params):
        # initialize algo
        algo = MonaiAlgo(**input_params)
        algo.initialize(extra={ExtraItems.CLIENT_NAME: "test_fl"})
        self.assertTrue(dist.get_rank() in (0, 1))

        # initialize model
        parser = ConfigParser()
        parser.read_config(
            [os.path.join(input_params["bundle_root"], x) for x in input_params["config_evaluate_filename"]]
        )
        parser.parse()
        network = parser.get_parsed_content("network")
        data = ExchangeObject(weights=network.state_dict())
        # test evaluate
        algo.evaluate(data=data, extra={})


if __name__ == "__main__":
    unittest.main()

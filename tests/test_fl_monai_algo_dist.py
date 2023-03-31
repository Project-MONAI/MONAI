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

from __future__ import annotations

import os
import unittest
from os.path import join as pathjoin

import torch.distributed as dist

from monai.bundle import ConfigParser, ConfigWorkflow
from monai.fl.client.monai_algo import MonaiAlgo
from monai.fl.utils.constants import ExtraItems
from monai.fl.utils.exchange_object import ExchangeObject
from tests.utils import DistCall, DistTestCase, SkipIfBeforePyTorchVersion, SkipIfNoModule, skip_if_no_cuda

_root_dir = os.path.abspath(pathjoin(os.path.dirname(__file__)))
_data_dir = pathjoin(_root_dir, "testing_data")


@SkipIfNoModule("ignite")
@SkipIfBeforePyTorchVersion((1, 11, 1))
class TestFLMonaiAlgo(DistTestCase):
    @DistCall(nnodes=1, nproc_per_node=2, init_method="no_init")
    @skip_if_no_cuda
    def test_train(self):
        config_file = [pathjoin(_data_dir, "config_fl_train.json"), pathjoin(_data_dir, "multi_gpu_train.json")]
        # initialize algo
        algo = MonaiAlgo(
            bundle_root=_data_dir,
            train_workflow=ConfigWorkflow(config_file=config_file, workflow="train"),
            config_evaluate_filename=None,
            config_filters_filename=pathjoin(_root_dir, "testing_data", "config_fl_filters.json"),
        )
        algo.initialize(extra={ExtraItems.CLIENT_NAME: "test_fl"})
        self.assertTrue(dist.get_rank() in (0, 1))

        # initialize model
        parser = ConfigParser()
        parser.read_config(config_file)
        parser.parse()
        network = parser.get_parsed_content("network")
        data = ExchangeObject(weights=network.state_dict())
        # test train
        algo.train(data=data, extra={})

    @DistCall(nnodes=1, nproc_per_node=2, init_method="no_init")
    @skip_if_no_cuda
    def test_evaluate(self):
        config_file = [pathjoin(_data_dir, "config_fl_evaluate.json"), pathjoin(_data_dir, "multi_gpu_evaluate.json")]
        # initialize algo
        algo = MonaiAlgo(
            bundle_root=_data_dir,
            config_train_filename=None,
            eval_workflow=ConfigWorkflow(config_file=config_file, workflow="train"),
            config_filters_filename=pathjoin(_data_dir, "config_fl_filters.json"),
        )
        algo.initialize(extra={ExtraItems.CLIENT_NAME: "test_fl"})
        self.assertTrue(dist.get_rank() in (0, 1))

        # initialize model
        parser = ConfigParser()
        parser.read_config(pathjoin(_data_dir, "config_fl_evaluate.json"))
        parser.parse()
        network = parser.get_parsed_content("network")
        data = ExchangeObject(weights=network.state_dict())
        # test evaluate
        algo.evaluate(data=data, extra={})


if __name__ == "__main__":
    unittest.main()

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

import torch
from parameterized import parameterized

from monai.apps.reconstruction.networks.nets.varnet import VariationalNetworkModel
from monai.networks import eval_mode
from tests.utils import test_script_save


# hyper-parameters object
class Args:
    def __init__(self, num_chans: int, sens_chans: int, num_cascades: int, gpus: list):
        self.num_chans = num_chans
        self.sens_chans = sens_chans
        self.num_cascades = num_cascades
        self.gpus = gpus

        if torch.cuda.is_available():
            devices = [torch.device(f"cuda:{i}") for i in gpus]
        else:
            devices = [torch.device("cpu")]
        self.devices = devices


TESTS = []
hparams = Args(num_chans=12, sens_chans=5, num_cascades=8, gpus=[0, 1])
TESTS.append([hparams, (1, 10, 30, 20, 2), (1, 30, 20)])  # batch=1
TESTS.append([hparams, (2, 10, 30, 20, 2), (2, 30, 20)])  # batch=2

hparams.devices = [torch.device("cpu")]
TESTS.append([hparams, (2, 10, 30, 20, 2), (2, 30, 20)])  # cpu device


class TestVarNet(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_shape(self, hparams, input_shape, expected_shape):
        net = VariationalNetworkModel(hparams)
        mask_shape = [1 for _ in input_shape]
        mask_shape[-2] = input_shape[-2]
        mask = torch.zeros(mask_shape)
        mask[..., mask_shape[-2] // 2 - 5 : mask_shape[-2] // 2 + 5, :] = 1

        with eval_mode(net):
            result = net(
                torch.randn(input_shape).to(hparams.devices[0]), mask.type(torch.ByteTensor).to(hparams.devices[0])
            )
        self.assertEqual(result.shape, expected_shape)

    """ Add this later
    @parameterized.expand(TESTS)
    def test_script(self,hparams, input_shape, expected_shape):
        net = VariationalNetworkModel(hparams)
        
        mask_shape = [1 for _ in input_shape]
        mask_shape[-2] = input_shape[-2]
        mask = torch.zeros(mask_shape)
        mask[..., mask_shape[-2]//2-5:mask_shape[-2]//2+5, :] = 1
        
        test_data = torch.randn(input_shape)
        
        test_script_save(net, [test_data.type(torch.FloatTensor), mask.type(torch.ByteTensor)])
    """


if __name__ == "__main__":
    unittest.main()

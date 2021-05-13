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

from monai.networks.nets import DenseNet, DenseNet121, NetworkFactory, UNet

TESTS = [
    [
        UNet,
        "UNet",
        {"dimensions": 3, "in_channels": 4, "out_channels": 2, "channels": (4, 8, 16, 32), "strides": (2, 4, 1)},
    ],
    [DenseNet, "DenseNet", {"spatial_dims": 2, "in_channels": 4, "out_channels": 2}],
]


class TestNetworkFactory(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_network_factory(self, class_name, name, args):
        net = NetworkFactory.get_constructor(name, **args)
        self.assertIsInstance(net, class_name)

    def test_fails(self):
        with self.assertRaises(KeyError):
            NetworkFactory["will_never_be_a_network"]

    def test_register_network(self):
        class_name = "BrandNewNetwork"

        class BrandNewNetwork(DenseNet121):
            pass

        NetworkFactory.add_factory_callable(class_name, BrandNewNetwork)
        kwargs = {"spatial_dims": 3, "in_channels": 3, "out_channels": 2}
        net = NetworkFactory.get_constructor(class_name.upper(), **kwargs)
        self.assertIsInstance(net, BrandNewNetwork)

    def test_num_networks(self):
        self.assertGreater(len(NetworkFactory.names), 30)


if __name__ == "__main__":
    unittest.main()

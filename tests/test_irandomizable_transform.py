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

import numpy as np

from monai.transforms.transform import IRandomizableTransform, RandomizableTransform


class InheritsInterface(IRandomizableTransform):
    pass


class InheritsImplementation(RandomizableTransform):

    def __call__(self, data):
        return data


class TestIRandomizableTransform(unittest.TestCase):

    def test_is_irandomizable(self):
        inst = InheritsInterface()
        self.assertIsInstance(inst, IRandomizableTransform)

    def test_set_random_state_default_impl(self):
        inst = InheritsInterface()
        with self.assertRaises(TypeError):
            inst.set_random_state(seed=0)

    def test_set_random_state_randomizable_transform(self):
        inst = InheritsImplementation()
        inst.set_random_state(0)

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

import unittest

import numpy as np
from monai.utils.utils_random_generator_adaptor import _LegacyRandomStateAdaptor

from monai.transforms.transform import Randomizable


class RandTest(Randomizable):
    def randomize(self, data=None):
        pass


class TestRandomizable(unittest.TestCase):
    def test_default(self):
        inst = RandTest()
        r1 = inst.R.random()
        self.assertTrue(isinstance(inst.R, _LegacyRandomStateAdaptor))
        inst.set_random_generator()
        r2 = inst.R.random()
        self.assertNotAlmostEqual(r1, r2)

    def test_seed(self):
        inst = RandTest()
        inst.set_random_generator(seed=123)
        self.assertAlmostEqual(inst.R.random(), 0.69646918)
        inst.set_random_generator(123)
        self.assertAlmostEqual(inst.R.random(), 0.69646918)

    def test_generator(self):
        inst = RandTest()
        inst_r = _LegacyRandomStateAdaptor(random_state=np.random.RandomState(123))
        inst.set_random_generator(generator=inst_r)
        self.assertAlmostEqual(inst.R.random(), 0.69646918)

    def test_legacy_default(self):
        inst = RandTest()
        r1 = inst.R.rand()
        self.assertTrue(isinstance(inst.R, _LegacyRandomStateAdaptor))
        inst.set_random_state()
        r2 = inst.R.rand()
        self.assertNotAlmostEqual(r1, r2)

    def test_legacy_seed(self):
        inst = RandTest()
        inst.set_random_state(seed=123)
        self.assertAlmostEqual(inst.R.rand(), 0.69646918)
        inst.set_random_state(123)
        self.assertAlmostEqual(inst.R.rand(), 0.69646918)

    def test_legacy_state(self):
        inst = RandTest()
        inst_r = np.random.RandomState(123)
        inst.set_random_state(state=inst_r)
        self.assertAlmostEqual(inst.R.rand(), 0.69646918)


if __name__ == "__main__":
    unittest.main()

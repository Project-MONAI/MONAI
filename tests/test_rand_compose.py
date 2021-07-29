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

import sys
import unittest

from monai.data import DataLoader, Dataset
from monai.transforms import AddChannel, Compose, RandCompose
from monai.transforms.transform import Randomizable
from monai.utils import set_determinism


class _RandXform(Randomizable):
    def randomize(self):
        self.val = self.R.random_sample()

    def __call__(self, img):
        self.randomize()
        return img + self.val


class TestRandCompose(unittest.TestCase):
    def test_non_dict_compose(self):
        def a(i):
            return i + "a"

        def b(i):
            return i + "b"

        c = RandCompose(prob=[1.0, 0.0, 0.0, 1.0], transforms=[a, b, a, b])
        self.assertEqual(c(""), "ab")

    def test_dict_compose(self):
        def a(d):
            d = dict(d)
            d["a"] += 1
            return d

        def b(d):
            d = dict(d)
            d["b"] += 1
            return d

        c = RandCompose(prob=[1.0, 0.0, 1.0, 0.0, 1.0], transforms=[a, b, a, b, a])
        self.assertDictEqual(c({"a": 0, "b": 0}), {"a": 3, "b": 0})

    def test_list_dict_compose(self):
        def a(d):  # transform to handle dict data
            d = dict(d)
            d["a"] += 1
            return d

        def b(d):  # transform to generate a batch list of data
            d = dict(d)
            d["b"] += 1
            d = [d] * 5
            return d

        def c(d):  # transform to handle dict data
            d = dict(d)
            d["c"] += 1
            return d

        transforms = RandCompose(prob=[1.0, 0.0, 1.0, 0.0, 1.0], transforms=[a, a, b, c, c])
        value = transforms({"a": 0, "b": 0, "c": 0})
        for item in value:
            self.assertDictEqual(item, {"a": 1, "b": 1, "c": 1})

    def test_non_dict_compose_with_unpack(self):
        def a(i, i2):
            return i + "a", i2 + "a2"

        def b(i, i2):
            return i + "b", i2 + "b2"

        c = RandCompose(prob=[1.0, 0.0, 0.0, 1.0], transforms=[a, b, a, b], map_items=False, unpack_items=True)
        self.assertEqual(c(("", "")), ("ab", "a2b2"))

    def test_list_non_dict_compose_with_unpack(self):
        def a(i, i2):
            return i + "a", i2 + "a2"

        def b(i, i2):
            return i + "b", i2 + "b2"

        c = RandCompose(prob=[1.0, 0.0, 0.0, 1.0], transforms=[a, b, a, b], unpack_items=True)
        self.assertEqual(c([("", ""), ("t", "t")]), [("ab", "a2b2"), ("tab", "ta2b2")])

    def test_list_dict_compose_no_map(self):
        def a(d):  # transform to handle dict data
            d = dict(d)
            d["a"] += 1
            return d

        def b(d):  # transform to generate a batch list of data
            d = dict(d)
            d["b"] += 1
            d = [d] * 5
            return d

        def c(d):  # transform to handle dict data
            d = [dict(di) for di in d]
            for di in d:
                di["c"] += 1
            return d

        transforms = RandCompose(prob=[1.0, 0.0, 1.0, 0.0, 1.0], transforms=[a, a, b, c, c], map_items=False)
        value = transforms({"a": 0, "b": 0, "c": 0})
        for item in value:
            self.assertDictEqual(item, {"a": 1, "b": 1, "c": 1})

    def test_random_compose(self):
        class _Acc(Randomizable):
            self.rand = 0.0

            def randomize(self, data=None):
                self.rand = self.R.rand()

            def __call__(self, data):
                self.randomize()
                return self.rand + data

        c = RandCompose(prob=0.5, transforms=[_Acc(), _Acc()])
        self.assertNotAlmostEqual(c(0), c(0))
        c.set_random_state(123)
        self.assertAlmostEqual(c(1), 1.61381597)
        c.set_random_state(456)
        c.randomize()
        self.assertAlmostEqual(c(1), 1.17330701)

    def test_data_loader(self):
        xform_1 = RandCompose(prob=0.5, transforms=[_RandXform(), _RandXform(), _RandXform()])
        train_ds = Dataset([1], transform=xform_1)

        set_determinism(seed=123)
        train_loader = DataLoader(train_ds, num_workers=0)
        out_1 = next(iter(train_loader))
        self.assertAlmostEqual(out_1.item(), 1.58704446)

        if sys.platform != "win32":  # skip multi-worker tests on win32
            train_loader = DataLoader(train_ds, num_workers=1)
            out_1 = next(iter(train_loader))
            self.assertAlmostEqual(out_1.item(), 1.15912328)

            train_loader = DataLoader(train_ds, num_workers=2)
            out_1 = next(iter(train_loader))
            self.assertAlmostEqual(out_1.item(), 1.65850210)
        set_determinism(None)

    def test_flatten_and_len(self):
        x = AddChannel()
        t1 = Compose([x, x, x, x, Compose([RandCompose(prob=[0.1, 0.2], transforms=[x, x]), x, x])])

        t2 = t1.flatten()
        # test length
        self.assertEqual(len(t1), 7)
        self.assertEqual(len(t2.transforms[4]), 2)


if __name__ == "__main__":
    unittest.main()

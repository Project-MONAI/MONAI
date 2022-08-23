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

import sys
import unittest

from monai.data import DataLoader, Dataset
from monai.transforms import AddChannel, Compose
from monai.transforms.transform import Randomizable
from monai.utils import set_determinism


class _RandXform(Randomizable):
    def randomize(self):
        self.val = self.R.random_sample()

    def __call__(self, __unused):
        self.randomize()
        return self.val


class TestCompose(unittest.TestCase):
    def test_empty_compose(self):
        c = Compose()
        i = 1
        self.assertEqual(c(i), 1)

    def test_non_dict_compose(self):
        def a(i):
            return i + "a"

        def b(i):
            return i + "b"

        c = Compose([a, b, a, b])
        self.assertEqual(c(""), "abab")

    def test_dict_compose(self):
        def a(d):
            d = dict(d)
            d["a"] += 1
            return d

        def b(d):
            d = dict(d)
            d["b"] += 1
            return d

        c = Compose([a, b, a, b, a])
        self.assertDictEqual(c({"a": 0, "b": 0}), {"a": 3, "b": 2})

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

        transforms = Compose([a, a, b, c, c])
        value = transforms({"a": 0, "b": 0, "c": 0})
        for item in value:
            self.assertDictEqual(item, {"a": 2, "b": 1, "c": 2})

    def test_non_dict_compose_with_unpack(self):
        def a(i, i2):
            return i + "a", i2 + "a2"

        def b(i, i2):
            return i + "b", i2 + "b2"

        c = Compose([a, b, a, b], map_items=False, unpack_items=True)
        self.assertEqual(c(("", "")), ("abab", "a2b2a2b2"))

    def test_list_non_dict_compose_with_unpack(self):
        def a(i, i2):
            return i + "a", i2 + "a2"

        def b(i, i2):
            return i + "b", i2 + "b2"

        c = Compose([a, b, a, b], unpack_items=True)
        self.assertEqual(c([("", ""), ("t", "t")]), [("abab", "a2b2a2b2"), ("tabab", "ta2b2a2b2")])

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

        transforms = Compose([a, a, b, c, c], map_items=False)
        value = transforms({"a": 0, "b": 0, "c": 0})
        for item in value:
            self.assertDictEqual(item, {"a": 2, "b": 1, "c": 2})

    def test_random_compose(self):
        class _Acc(Randomizable):
            self.rand = 0.0

            def randomize(self, data=None):
                self.rand = self.R.rand()

            def __call__(self, data):
                self.randomize()
                return self.rand + data

        c = Compose([_Acc(), _Acc()])
        self.assertNotAlmostEqual(c(0), c(0))
        c.set_random_state(123)
        self.assertAlmostEqual(c(1), 1.61381597)
        c.set_random_state(223)
        c.randomize()
        self.assertAlmostEqual(c(1), 1.90734751)

    def test_randomize_warn(self):
        class _RandomClass(Randomizable):
            def randomize(self, foo1, foo2):
                pass

            def __call__(self, data):
                pass

        c = Compose([_RandomClass(), _RandomClass()])
        with self.assertWarns(Warning):
            c.randomize()

    def test_err_msg(self):
        transforms = Compose([abs, AddChannel(), round], log_stats=False)
        with self.assertRaisesRegex(Exception, "AddChannel"):
            transforms(42.1)

    def test_data_loader(self):
        xform_1 = Compose([_RandXform()])
        train_ds = Dataset([1], transform=xform_1)

        xform_1.set_random_state(123)
        out_1 = train_ds[0]
        self.assertAlmostEqual(out_1, 0.2045649)

        set_determinism(seed=123)
        train_loader = DataLoader(train_ds, num_workers=0)
        out_1 = next(iter(train_loader))
        self.assertAlmostEqual(out_1.cpu().item(), 0.84291356)

        if sys.platform != "win32":  # skip multi-worker tests on win32
            train_loader = DataLoader(train_ds, num_workers=1)
            out_1 = next(iter(train_loader))
            self.assertAlmostEqual(out_1.cpu().item(), 0.180814653)

            train_loader = DataLoader(train_ds, num_workers=2)
            out_1 = next(iter(train_loader))
            self.assertAlmostEqual(out_1.cpu().item(), 0.04293707)
        set_determinism(None)

    def test_data_loader_2(self):
        set_determinism(seed=123)
        xform_2 = Compose([_RandXform(), _RandXform()])
        train_ds = Dataset([1], transform=xform_2)

        out_2 = train_ds[0]
        self.assertAlmostEqual(out_2, 0.4092510)

        train_loader = DataLoader(train_ds, num_workers=0)
        out_2 = next(iter(train_loader))
        self.assertAlmostEqual(out_2.cpu().item(), 0.7858843729)

        if sys.platform != "win32":  # skip multi-worker tests on win32
            train_loader = DataLoader(train_ds, num_workers=1)
            out_2 = next(iter(train_loader))
            self.assertAlmostEqual(out_2.cpu().item(), 0.305763411)

            train_loader = DataLoader(train_ds, num_workers=2)
            out_1 = next(iter(train_loader))
            self.assertAlmostEqual(out_1.cpu().item(), 0.131966779)
        set_determinism(None)

    def test_flatten_and_len(self):
        x = AddChannel()
        t1 = Compose([x, x, x, x, Compose([Compose([x, x]), x, x])])

        t2 = t1.flatten()
        for t in t2.transforms:
            self.assertNotIsInstance(t, Compose)

        # test len
        self.assertEqual(len(t1), 8)

    def test_backwards_compatible_imports(self):
        from monai.transforms.compose import MapTransform, RandomizableTransform, Transform  # noqa: F401


if __name__ == "__main__":
    unittest.main()

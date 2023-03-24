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

import sys
import unittest
from copy import deepcopy

from parameterized import parameterized

import numpy as np
import torch

from monai.data import DataLoader, Dataset
from monai.transforms import AddChannel, Compose, Flip, Rotate90, Zoom, NormalizeIntensity, Rotate, Rotated
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

        transforms = [a, b, a, b, a]
        data = {"a": 0, "b": 0}
        expected = {"a": 3, "b": 2}

        self.assertDictEqual(Compose(transforms)(data), expected)
        self.assertDictEqual(Compose.execute(data, transforms), expected)

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

        transforms = [a, a, b, c, c]
        data = {"a": 0, "b": 0, "c": 0}
        expected = {"a": 2, "b": 1, "c": 2}
        value = Compose(transforms)(data)
        for item in value:
            self.assertDictEqual(item, expected)
        value = Compose.execute(data, transforms)
        for item in value:
            self.assertDictEqual(item, expected)

    def test_non_dict_compose_with_unpack(self):
        def a(i, i2):
            return i + "a", i2 + "a2"

        def b(i, i2):
            return i + "b", i2 + "b2"

        transforms = [a, b, a, b]
        data = ("", "")
        expected = ("abab", "a2b2a2b2")
        self.assertEqual(Compose(transforms, map_items=False, unpack_items=True)(data), expected)
        self.assertEqual(Compose.execute(data, transforms, map_items=False, unpack_items=True), expected)

    def test_list_non_dict_compose_with_unpack(self):
        def a(i, i2):
            return i + "a", i2 + "a2"

        def b(i, i2):
            return i + "b", i2 + "b2"

        transforms = [a, b, a, b]
        data = [("", ""), ("t", "t")]
        expected = [("abab", "a2b2a2b2"), ("tabab", "ta2b2a2b2")]
        self.assertEqual(Compose(transforms, unpack_items=True)(data), expected)
        self.assertEqual(Compose.execute(data, transforms, unpack_items=True), expected)

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

        transforms = [a, a, b, c, c]
        data = {"a": 0, "b": 0, "c": 0}
        expected = {"a": 2, "b": 1, "c": 2}
        value = Compose(transforms, map_items=False)(data)
        for item in value:
            self.assertDictEqual(item, expected)
        value = Compose.execute(data, transforms, map_items=False)
        for item in value:
            self.assertDictEqual(item, expected)


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
        self.assertAlmostEqual(out_1.cpu().item(), 0.0409280)

        if sys.platform != "win32":  # skip multi-worker tests on win32
            train_loader = DataLoader(train_ds, num_workers=1)
            out_1 = next(iter(train_loader))
            self.assertAlmostEqual(out_1.cpu().item(), 0.78663897075)

            train_loader = DataLoader(train_ds, num_workers=2)
            out_1 = next(iter(train_loader))
            self.assertAlmostEqual(out_1.cpu().item(), 0.785907334)
        set_determinism(None)

    def test_data_loader_2(self):
        set_determinism(seed=123)
        xform_2 = Compose([_RandXform(), _RandXform()])
        train_ds = Dataset([1], transform=xform_2)

        out_2 = train_ds[0]
        self.assertAlmostEqual(out_2, 0.4092510)

        train_loader = DataLoader(train_ds, num_workers=0)
        out_2 = next(iter(train_loader))
        self.assertAlmostEqual(out_2.cpu().item(), 0.98921915918)

        if sys.platform != "win32":  # skip multi-worker tests on win32
            train_loader = DataLoader(train_ds, num_workers=1)
            out_2 = next(iter(train_loader))
            self.assertAlmostEqual(out_2.cpu().item(), 0.32985207)

            train_loader = DataLoader(train_ds, num_workers=2)
            out_1 = next(iter(train_loader))
            self.assertAlmostEqual(out_1.cpu().item(), 0.28602141572)
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


TEST_COMPOSE_EXECUTE_TEST_CASES = [
    [None, tuple()],
    [None, (Rotate(np.pi/8),)],
    [None, (Flip(0), Flip(1), Rotate90(1), Zoom(0.8), NormalizeIntensity())],
    [('a',), (Rotated(('a',), np.pi/8),)],
]


class TestComposeExecute(unittest.TestCase):

    @parameterized.expand(TEST_COMPOSE_EXECUTE_TEST_CASES)
    def test_compose_execute_equivalence(self, keys, pipeline):

        if keys is None:
            data = torch.unsqueeze(torch.tensor(np.arange(24*32).reshape(24, 32)), axis=0)
        else:
            data = {}
            for i_k, k in enumerate(keys):
                data[k] = torch.unsqueeze(torch.tensor(np.arange(24*32)).reshape(24, 32) + i_k * 768,
                                          axis=0)

        expected = Compose(deepcopy(pipeline))(data)

        for cutoff in range(len(pipeline)):

            c = Compose(deepcopy(pipeline))
            actual = c(c(data, end=cutoff), start=cutoff)
            if isinstance(actual, dict):
                for k in actual.keys():
                    self.assertTrue(torch.allclose(expected[k], actual[k]))
            else:
                self.assertTrue(torch.allclose(expected, actual))

            p = deepcopy(pipeline)
            actual = Compose.execute(
                Compose.execute(data, p, start=0, end=cutoff), p, start=cutoff)
            if isinstance(actual, dict):
                for k in actual.keys():
                    self.assertTrue(torch.allclose(expected[k], actual[k]))
            else:
                self.assertTrue(torch.allclose(expected, actual))


class TestOps:

    @staticmethod
    def concat(value):
        def _inner(data):
            return data + value

        return _inner

    @staticmethod
    def concatd(value):
        def _inner(data):
            return {k: v + value for k, v in data.items()}

        return _inner

    @staticmethod
    def concata(value):
        def _inner(data1, data2):
            return data1 + value, data2 + value

        return _inner


TEST_COMPOSE_EXECUTE_FLAG_TEST_CASES = [
    [{}, ("",), (TestOps.concat('a'), TestOps.concat('b'))],
    [{"unpack_items": True}, ("x", "y"), (TestOps.concat('a'), TestOps.concat('b'))],
    [{"map_items": False}, {"x": "1", "y": "2"}, (TestOps.concatd('a'), TestOps.concatd('b'))],
    [{"unpack_items": True, "map_items": False}, ("x", "y"), (TestOps.concata('a'), TestOps.concata('b'))],
]


class TestComposeExecuteWithFlags(unittest.TestCase):

    @parameterized.expand(TEST_COMPOSE_EXECUTE_FLAG_TEST_CASES)
    def test_compose_execute_equivalence_with_flags(self, flags, data, pipeline):
        expected = Compose(pipeline, **flags)(data)

        for cutoff in range(len(pipeline)):

            c = Compose(deepcopy(pipeline), **flags)
            actual = c(c(data, end=cutoff), start=cutoff)
            if isinstance(actual, dict):
                for k in actual.keys():
                    self.assertEqual(expected[k], actual[k])
            else:
                self.assertTrue(expected, actual)

            p = deepcopy(pipeline)
            actual = Compose.execute(
                Compose.execute(data, p, start=0, end=cutoff, **flags), p, start=cutoff, **flags)
            if isinstance(actual, dict):
                for k in actual.keys():
                    self.assertTrue(expected[k], actual[k])
            else:
                self.assertTrue(expected, actual)


if __name__ == "__main__":
    unittest.main()

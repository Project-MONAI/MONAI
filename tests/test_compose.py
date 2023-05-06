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

import logging
import sys
import unittest
from copy import deepcopy
from io import StringIO

import numpy as np
import torch
from parameterized import parameterized

import monai.transforms as mt
from monai.data import DataLoader, Dataset
from monai.transforms.compose import ExecutionOptions, execute_compose
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
        c = mt.Compose()
        i = 1
        self.assertEqual(c(i), 1)

    def test_non_dict_compose(self):
        def a(i):
            return i + "a"

        def b(i):
            return i + "b"

        c = mt.Compose([a, b, a, b])
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

        self.assertDictEqual(mt.Compose(transforms)(data), expected)
        self.assertDictEqual(execute_compose(data, transforms), expected)

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
        value = mt.Compose(transforms)(data)
        for item in value:
            self.assertDictEqual(item, expected)
        value = execute_compose(data, transforms)
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
        self.assertEqual(mt.Compose(transforms, map_items=False, unpack_items=True)(data), expected)
        self.assertEqual(execute_compose(data, transforms, map_items=False, unpack_items=True), expected)

    def test_list_non_dict_compose_with_unpack(self):
        def a(i, i2):
            return i + "a", i2 + "a2"

        def b(i, i2):
            return i + "b", i2 + "b2"

        transforms = [a, b, a, b]
        data = [("", ""), ("t", "t")]
        expected = [("abab", "a2b2a2b2"), ("tabab", "ta2b2a2b2")]
        self.assertEqual(mt.Compose(transforms, unpack_items=True)(data), expected)
        self.assertEqual(execute_compose(data, transforms, unpack_items=True), expected)

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
        value = mt.Compose(transforms, map_items=False)(data)
        for item in value:
            self.assertDictEqual(item, expected)
        value = execute_compose(data, transforms, map_items=False)
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

        c = mt.Compose([_Acc(), _Acc()])
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

        c = mt.Compose([_RandomClass(), _RandomClass()])
        with self.assertWarns(Warning):
            c.randomize()

    def test_err_msg(self):
        transforms = mt.Compose([abs, mt.AddChannel(), round])
        with self.assertRaisesRegex(Exception, "AddChannel"):
            transforms(42.1)

    def test_data_loader(self):
        xform_1 = mt.Compose([_RandXform()])
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
        xform_2 = mt.Compose([_RandXform(), _RandXform()])
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
        x = mt.AddChannel()
        t1 = mt.Compose([x, x, x, x, mt.Compose([mt.Compose([x, x]), x, x])])

        t2 = t1.flatten()
        for t in t2.transforms:
            self.assertNotIsInstance(t, mt.Compose)

        # test len
        self.assertEqual(len(t1), 8)

    def test_backwards_compatible_imports(self):
        from monai.transforms.transform import MapTransform, RandomizableTransform, Transform  # noqa: F401


TEST_COMPOSE_EXECUTE_TEST_CASES = [
    [None, tuple()],
    [None, (mt.Rotate(np.pi / 8),)],
    [None, (mt.Flip(0), mt.Flip(1), mt.Rotate90(1), mt.Zoom(0.8), mt.NormalizeIntensity())],
    [("a",), (mt.Rotated(("a",), np.pi / 8),)],
]


class TestComposeExecute(unittest.TestCase):
    @staticmethod
    def data_from_keys(keys):
        if keys is None:
            data = torch.unsqueeze(torch.tensor(np.arange(12 * 16).reshape(12, 16)), dim=0)
        else:
            data = {}
            for i_k, k in enumerate(keys):
                data[k] = torch.unsqueeze(torch.tensor(np.arange(12 * 16)).reshape(12, 16) + i_k * 192, dim=0)
        return data

    @parameterized.expand(TEST_COMPOSE_EXECUTE_TEST_CASES)
    def test_compose_execute_equivalence(self, keys, pipeline):
        data = self.data_from_keys(keys)

        expected = mt.Compose(deepcopy(pipeline))(data)

        for cutoff in range(len(pipeline)):
            c = mt.Compose(deepcopy(pipeline))
            actual = c(c(data, end=cutoff), start=cutoff)
            if isinstance(actual, dict):
                for k in actual.keys():
                    self.assertTrue(torch.allclose(expected[k], actual[k]))
            else:
                self.assertTrue(torch.allclose(expected, actual))

            p = deepcopy(pipeline)
            actual = execute_compose(execute_compose(data, p, start=0, end=cutoff), p, start=cutoff)
            if isinstance(actual, dict):
                for k in actual.keys():
                    self.assertTrue(torch.allclose(expected[k], actual[k]))
            else:
                self.assertTrue(torch.allclose(expected, actual))

    @parameterized.expand(TEST_COMPOSE_EXECUTE_TEST_CASES)
    def test_compose_execute_bad_start_param(self, keys, pipeline):
        data = self.data_from_keys(keys)

        with self.assertRaises(ValueError):
            c = mt.Compose(deepcopy(pipeline))
            c(data, start=None)

        with self.assertRaises(ValueError):
            execute_compose(data, deepcopy(pipeline), start=None)

        with self.assertRaises(ValueError):
            c = mt.Compose(deepcopy(pipeline))
            c(data, start=-1)

        with self.assertRaises(ValueError):
            execute_compose(data, deepcopy(pipeline), start=-1)

    @parameterized.expand(TEST_COMPOSE_EXECUTE_TEST_CASES)
    def test_compose_execute_negative_range(self, keys, pipeline):
        data = self.data_from_keys(keys)

        with self.assertRaises(ValueError):
            c = mt.Compose(deepcopy(pipeline))
            c(data, start=2, end=1)

        with self.assertRaises(ValueError):
            execute_compose(data, deepcopy(pipeline), start=2, end=1)

    @parameterized.expand(TEST_COMPOSE_EXECUTE_TEST_CASES)
    def test_compose_execute_bad_end_param(self, keys, pipeline):
        data = self.data_from_keys(keys)

        with self.assertRaises(ValueError):
            c = mt.Compose(deepcopy(pipeline))
            c(data, end=len(pipeline) + 1)

        with self.assertRaises(ValueError):
            execute_compose(data, deepcopy(pipeline), end=len(pipeline) + 1)

    @parameterized.expand(TEST_COMPOSE_EXECUTE_TEST_CASES)
    def test_compose_execute_empty_range(self, keys, pipeline):
        data = self.data_from_keys(keys)

        c = mt.Compose(deepcopy(pipeline))
        for i in range(len(pipeline)):
            result = c(data, start=i, end=i)
            self.assertIs(data, result)

    @parameterized.expand(TEST_COMPOSE_EXECUTE_TEST_CASES)
    def test_compose_with_logger_name(self, keys, pipeline):
        data = self.data_from_keys(keys)

        c = mt.Compose(deepcopy(pipeline), logger_name="a_logger_name")
        c(data)


TEST_COMPOSE_EXECUTE_LOGGING_TEST_CASES = [
    [
        None,
        (mt.Flip(0), mt.Spacing((1.2, 1.2)), mt.Flip(1), mt.Rotate90(1), mt.Zoom(0.8), mt.NormalizeIntensity()),
        False,
        (
            "INFO - Apply pending transforms - lazy: False, pending: 0, "
            "upcoming 'Flip', transform.lazy: False\n"
            "INFO - Apply pending transforms - lazy: False, pending: 0, "
            "upcoming 'Spacing', transform.lazy: False\n"
            "INFO - Apply pending transforms - lazy: False, pending: 0, "
            "upcoming 'Flip', transform.lazy: False\n"
            "INFO - Apply pending transforms - lazy: False, pending: 0, "
            "upcoming 'Rotate90', transform.lazy: False\n"
            "INFO - Apply pending transforms - lazy: False, pending: 0, "
            "upcoming 'Zoom', transform.lazy: False\n"
            "INFO - Apply pending transforms - lazy: False, pending: 0, "
            "upcoming 'NormalizeIntensity', transform is not lazy\n"
        ),
    ],
    [
        None,
        (
            mt.Flip(0, lazy=True),
            mt.Spacing((1.2, 1.2), lazy=True),
            mt.Flip(1, lazy=True),
            mt.Rotate90(1),
            mt.Zoom(0.8, lazy=True),
            mt.NormalizeIntensity(),
        ),
        None,
        (
            "INFO - Accumulate pending transforms - lazy: None, pending: 0, "
            "upcoming 'Flip', transform.lazy: True\n"
            "INFO - Accumulate pending transforms - lazy: None, pending: 1, "
            "upcoming 'Spacing', transform.lazy: True\n"
            "INFO - Accumulate pending transforms - lazy: None, pending: 2, "
            "upcoming 'Flip', transform.lazy: True\n"
            "INFO - Apply pending transforms - lazy: None, pending: 3, "
            "upcoming 'Rotate90', transform.lazy: False\n"
            "INFO - Pending transforms applied: applied_operations: 3\n"
            "INFO - Accumulate pending transforms - lazy: None, pending: 0, "
            "upcoming 'Zoom', transform.lazy: True\n"
            "INFO - Apply pending transforms - lazy: None, pending: 1, "
            "upcoming 'NormalizeIntensity', transform is not lazy\n"
            "INFO - Pending transforms applied: applied_operations: 5\n"
        ),
    ],
    [
        None,
        (mt.Flip(0), mt.Spacing((1.2, 1.2)), mt.Flip(1), mt.Rotate90(1), mt.Zoom(0.8), mt.NormalizeIntensity()),
        True,
        (
            "INFO - Accumulate pending transforms - lazy: True, pending: 0, "
            "upcoming 'Flip', transform.lazy: False (overridden)\n"
            "INFO - Accumulate pending transforms - lazy: True, pending: 1, "
            "upcoming 'Spacing', transform.lazy: False (overridden)\n"
            "INFO - Accumulate pending transforms - lazy: True, pending: 2, "
            "upcoming 'Flip', transform.lazy: False (overridden)\n"
            "INFO - Accumulate pending transforms - lazy: True, pending: 3, "
            "upcoming 'Rotate90', transform.lazy: False (overridden)\n"
            "INFO - Accumulate pending transforms - lazy: True, pending: 4, "
            "upcoming 'Zoom', transform.lazy: False (overridden)\n"
            "INFO - Apply pending transforms - lazy: True, pending: 5, "
            "upcoming 'NormalizeIntensity', transform is not lazy\n"
            "INFO - Pending transforms applied: applied_operations: 5\n"
        ),
    ],
    [
        ("a", "b"),
        (
            mt.Flipd(("a", "b"), 0),
            mt.Spacingd(("a", "b"), 1.2),
            mt.Rotate90d(("a", "b"), 1),
            mt.NormalizeIntensityd(("a",)),
        ),
        True,
        (
            "INFO - Accumulate pending transforms - lazy mode: True, key: 'a', pending: 0, "
            "upcoming 'Flipd', transform.lazy: False (overridden)\n"
            "INFO - Accumulate pending transforms - lazy mode: True, key: 'b', pending: 0, "
            "upcoming 'Flipd', transform.lazy: False (overridden)\n"
            "INFO - Accumulate pending transforms - lazy mode: True, key: 'a', pending: 1, "
            "upcoming 'Spacingd', transform.lazy: False (overridden)\n"
            "INFO - Accumulate pending transforms - lazy mode: True, key: 'b', pending: 1, "
            "upcoming 'Spacingd', transform.lazy: False (overridden)\n"
            "INFO - Accumulate pending transforms - lazy mode: True, key: 'a', pending: 2, "
            "upcoming 'Rotate90d', transform.lazy: False (overridden)\n"
            "INFO - Accumulate pending transforms - lazy mode: True, key: 'b', pending: 2, "
            "upcoming 'Rotate90d', transform.lazy: False (overridden)\n"
            "INFO - Apply pending transforms - lazy mode: True, key: 'a', pending: 3, "
            "upcoming 'NormalizeIntensityd', transform is not lazy\n"
            "INFO - Pending transforms applied: key: 'a', applied_operations: 3\n"
            "INFO - Pending transforms applied: key: 'b', applied_operations: 3\n"
        ),
    ],
    [
        ("a", "b"),
        (
            mt.Flipd(keys="a", spatial_axis=0),
            mt.Rotate90d(keys="b", k=1, allow_missing_keys=True),
            mt.Zoomd(keys=("a", "b"), zoom=0.8, allow_missing_keys=True),
            mt.Spacingd(keys="a", pixdim=1.2),
        ),
        True,
        (
            "INFO - Accumulate pending transforms - lazy mode: True, key: 'a', pending: 0, "
            "upcoming 'Flipd', transform.lazy: False (overridden)\n"
            "INFO - Accumulate pending transforms - lazy mode: True, key: 'b', pending: 0, "
            "upcoming 'Rotate90d', transform.lazy: False (overridden)\n"
            "INFO - Accumulate pending transforms - lazy mode: True, key: 'a', pending: 1, "
            "upcoming 'Zoomd', transform.lazy: False (overridden)\n"
            "INFO - Accumulate pending transforms - lazy mode: True, key: 'b', pending: 1, "
            "upcoming 'Zoomd', transform.lazy: False (overridden)\n"
            "INFO - Accumulate pending transforms - lazy mode: True, key: 'a', pending: 2, "
            "upcoming 'Spacingd', transform.lazy: False (overridden)\n"
            "INFO - Pending transforms applied: key: 'a', applied_operations: 3\n"
            "INFO - Pending transforms applied: key: 'b', applied_operations: 2\n"
        ),
    ],
    [
        None,
        (
            mt.Flip(0),
            mt.Spacing((1.2, 1.2)),
            mt.Flip(1),
            mt.ApplyPending(),
            mt.Rotate90(1),
            mt.Zoom(0.8),
            mt.NormalizeIntensity(),
        ),
        False,
        (
            "INFO - Apply pending transforms - lazy: False, pending: 0, "
            "upcoming 'Flip', transform.lazy: False\n"
            "INFO - Apply pending transforms - lazy: False, pending: 0, "
            "upcoming 'Spacing', transform.lazy: False\n"
            "INFO - Apply pending transforms - lazy: False, pending: 0, "
            "upcoming 'Flip', transform.lazy: False\n"
            "INFO - Apply pending transforms - lazy: False, pending: 0, "
            "upcoming 'ApplyPending', transform is not lazy\n"
            "INFO - Apply pending transforms - lazy: False, pending: 0, "
            "upcoming 'Rotate90', transform.lazy: False\n"
            "INFO - Apply pending transforms - lazy: False, pending: 0, "
            "upcoming 'Zoom', transform.lazy: False\n"
            "INFO - Apply pending transforms - lazy: False, pending: 0, "
            "upcoming 'NormalizeIntensity', transform is not lazy\n"
        ),
    ],
    [
        None,
        (
            mt.Flip(0),
            mt.Spacing((1.2, 1.2)),
            mt.Flip(1),
            mt.ApplyPending(),
            mt.Rotate90(1),
            mt.Zoom(0.8),
            mt.NormalizeIntensity(),
        ),
        True,
        (
            "INFO - Accumulate pending transforms - lazy: True, pending: 0, "
            "upcoming 'Flip', transform.lazy: False (overridden)\n"
            "INFO - Accumulate pending transforms - lazy: True, pending: 1, "
            "upcoming 'Spacing', transform.lazy: False (overridden)\n"
            "INFO - Accumulate pending transforms - lazy: True, pending: 2, "
            "upcoming 'Flip', transform.lazy: False (overridden)\n"
            "INFO - Apply pending transforms - lazy: True, pending: 3, "
            "upcoming 'ApplyPending', transform is not lazy\n"
            "INFO - Pending transforms applied: applied_operations: 3\n"
            "INFO - Accumulate pending transforms - lazy: True, pending: 0, "
            "upcoming 'Rotate90', transform.lazy: False (overridden)\n"
            "INFO - Accumulate pending transforms - lazy: True, pending: 1, "
            "upcoming 'Zoom', transform.lazy: False (overridden)\n"
            "INFO - Apply pending transforms - lazy: True, pending: 2, "
            "upcoming 'NormalizeIntensity', transform is not lazy\n"
            "INFO - Pending transforms applied: applied_operations: 5\n"
        ),
    ],
    [
        ("a", "b"),
        (
            mt.Flipd(keys="a", spatial_axis=0),
            mt.Rotate90d(keys="b", k=1, allow_missing_keys=True),
            mt.ApplyPendingd(keys=("a", "b")),
            mt.Zoomd(keys=("a", "b"), zoom=0.8, allow_missing_keys=True),
            mt.Spacingd(keys="a", pixdim=1.2),
        ),
        True,
        (
            "INFO - Accumulate pending transforms - lazy mode: True, key: 'a', pending: 0, "
            "upcoming 'Flipd', transform.lazy: False (overridden)\n"
            "INFO - Accumulate pending transforms - lazy mode: True, key: 'b', pending: 0, "
            "upcoming 'Rotate90d', transform.lazy: False (overridden)\n"
            "INFO - Apply pending transforms - lazy mode: True, key: 'a', pending: 1, "
            "upcoming 'ApplyPendingd', transform is not lazy\n"
            "INFO - Apply pending transforms - lazy mode: True, key: 'b', pending: 1, "
            "upcoming 'ApplyPendingd', transform is not lazy\n"
            "INFO - Pending transforms applied: key: 'a', applied_operations: 1\n"
            "INFO - Pending transforms applied: key: 'b', applied_operations: 1\n"
            "INFO - Accumulate pending transforms - lazy mode: True, key: 'a', pending: 0, "
            "upcoming 'Zoomd', transform.lazy: False (overridden)\n"
            "INFO - Accumulate pending transforms - lazy mode: True, key: 'b', pending: 0, "
            "upcoming 'Zoomd', transform.lazy: False (overridden)\n"
            "INFO - Accumulate pending transforms - lazy mode: True, key: 'a', pending: 1, "
            "upcoming 'Spacingd', transform.lazy: False (overridden)\n"
            "INFO - Pending transforms applied: key: 'a', applied_operations: 3\n"
            "INFO - Pending transforms applied: key: 'b', applied_operations: 2\n"
        ),
    ],
]


class TestComposeExecuteWithLogging(unittest.TestCase):
    @staticmethod
    def data_from_keys(keys):
        if keys is None:
            data = torch.unsqueeze(torch.tensor(np.arange(12 * 16).reshape(12, 16)), dim=0)
        else:
            data = {}
            for i_k, k in enumerate(keys):
                data[k] = torch.unsqueeze(torch.tensor(np.arange(12 * 16)).reshape(12, 16) + i_k * 192, dim=0)
        return data

    @parameterized.expand(TEST_COMPOSE_EXECUTE_LOGGING_TEST_CASES)
    def test_compose_with_logging(self, keys, pipeline, lazy, expected):
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        formatter = logging.Formatter("%(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger = logging.getLogger("a_logger_name")
        logger.setLevel(logging.INFO)
        while len(logger.handlers) > 0:
            logger.removeHandler(logger.handlers[-1])
        logger.addHandler(handler)

        data = self.data_from_keys(keys)
        c = mt.Compose(deepcopy(pipeline), lazy=lazy, logger_name="a_logger_name")
        c(data)

        handler.flush()
        actual = stream.getvalue()
        self.assertEqual(actual, expected)


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
    [{}, ("",), (TestOps.concat("a"), TestOps.concat("b"))],
    [{"unpack_items": True}, ("x", "y"), (TestOps.concat("a"), TestOps.concat("b"))],
    [{"map_items": False}, {"x": "1", "y": "2"}, (TestOps.concatd("a"), TestOps.concatd("b"))],
    [{"unpack_items": True, "map_items": False}, ("x", "y"), (TestOps.concata("a"), TestOps.concata("b"))],
]


class TestComposeExecuteWithFlags(unittest.TestCase):
    @parameterized.expand(TEST_COMPOSE_EXECUTE_FLAG_TEST_CASES)
    def test_compose_execute_equivalence_with_flags(self, flags, data, pipeline):
        @staticmethod
        def data_from_keys(keys):
            if keys is None:
                data = torch.unsqueeze(torch.tensor(np.arange(24 * 32).reshape(24, 32)), dim=0)
            else:
                data = {}
                for i_k, k in enumerate(keys):
                    data[k] = torch.unsqueeze(torch.tensor(np.arange(24 * 32)).reshape(24, 32) + i_k * 768, dim=0)
            return data

        expected = mt.Compose(pipeline, **flags)(data)

        for cutoff in range(len(pipeline)):
            c = mt.Compose(deepcopy(pipeline), **flags)
            actual = c(c(data, end=cutoff), start=cutoff)
            if isinstance(actual, dict):
                for k in actual.keys():
                    self.assertEqual(expected[k], actual[k])
            else:
                self.assertTrue(expected, actual)

            p = deepcopy(pipeline)
            actual = execute_compose(execute_compose(data, p, start=0, end=cutoff, **flags), p, start=cutoff, **flags)
            if isinstance(actual, dict):
                for k in actual.keys():
                    self.assertTrue(expected[k], actual[k])
            else:
                self.assertTrue(expected, actual)


TEST_LAZY_COMPOSE_PIPELINE_FIX_CASES = [
    [(mt.Flip(0), mt.Flip(1), mt.Rotate90(1), mt.Zoom(0.8), mt.NormalizeIntensity())]
]


class TestLazyComposePipelineFixes(unittest.TestCase):
    @parameterized.expand(TEST_LAZY_COMPOSE_PIPELINE_FIX_CASES)
    def test_lazy_compose_pipeline_fixes(self, pipeline):
        data = torch.unsqueeze(torch.tensor(np.arange(12 * 16).reshape(12, 16)), dim=0)
        c = mt.Compose(deepcopy(pipeline), lazy=True)
        _ = c(data)


class TNonLazy(mt.Transform):
    def __init__(self, tag):
        self.tag = tag

    def __call__(self, data):
        return data


class TLazy(mt.LazyTransform):
    def __init__(self, tag, lazy):
        super().__init__(lazy)
        self.tag = tag

    def __call__(self, data):
        return data


class TApplyPending(mt.ApplyPending):
    def __init__(self, tag):
        super().__init__()
        self.tag = tag


TRANSFORM_REORDERING_TEST_CASES = [
    (
        [TNonLazy("a"), TLazy("lb", True), TLazy("lc", True), TApplyPending("ad"), TLazy("le", True), TNonLazy("f")],
        {"reorder": "lazy_last"},
        ["a", "lb", "lc", "ad", "f", "le"],
        ["a", "lb", "lc", "ad", "f", "le"],
    ),
    (
        [TNonLazy("a"), TLazy("lb", True), TLazy("lc", True), TApplyPending("ad"), TLazy("le", False), TNonLazy("f")],
        {"reorder": "lazy_last"},
        ["a", "lb", "lc", "ad", "le", "f"],
        ["a", "lb", "lc", "ad", "f", "le"],
    ),
    (
        [TLazy("la", True), TNonLazy("b"), TLazy("lc", True), TApplyPending("ad"), TLazy("le", True), TNonLazy("f")],
        {"reorder": "lazy_last"},
        ["b", "la", "lc", "ad", "f", "le"],
        ["b", "la", "lc", "ad", "f", "le"],
    ),
    (
        [TLazy("la", False), TNonLazy("b"), TLazy("lc", True), TApplyPending("ad"), TLazy("le", True), TNonLazy("f")],
        {"reorder": "lazy_last"},
        ["la", "b", "lc", "ad", "f", "le"],
        ["b", "la", "lc", "ad", "f", "le"],
    ),
    (
        [TLazy("la", True), TNonLazy("b"), TLazy("lc", True), TApplyPending("ad"), TLazy("le", True), TNonLazy("f")],
        {"reorder": "lazy_last"},
        ["b", "la", "lc", "ad", "f", "le"],
        ["b", "la", "lc", "ad", "f", "le"],
    ),
    (
        [TNonLazy("a"), TLazy("lb", True), TLazy("lc", True), TApplyPending("ad"), TLazy("le", False), TNonLazy("f")],
        {"reorder": "lazy_last"},
        ["a", "lb", "lc", "ad", "le", "f"],
        ["a", "lb", "lc", "ad", "f", "le"],
    ),
    (
        [TLazy("la", True), TLazy("lb", True), TNonLazy("c"), TApplyPending("ad"), TApplyPending("ae"), TNonLazy("f")],
        {"reorder": "lazy_last"},
        ["c", "la", "lb", "ad", "ae", "f"],
        ["c", "la", "lb", "ad", "ae", "f"],
    ),
    (
        [TNonLazy("a"), TLazy("lb", True), TLazy("lc", True), TApplyPending("ad"), TLazy("le", True), TNonLazy("f")],
        {"reorder": "lazy_last"},
        ["a", "lb", "lc", "ad", "f", "le"],
        ["a", "lb", "lc", "ad", "f", "le"],
    ),
    (
        [
            TNonLazy("a"),
            TLazy("lb", True),
            TLazy("lc", True),
            TApplyPending("ad"),
            TLazy("le", True),
            TApplyPending("af"),
            TApplyPending("ag"),
        ],
        {"reorder": "lazy_last"},
        ["a", "lb", "lc", "ad", "le", "af", "ag"],
        ["a", "lb", "lc", "ad", "le", "af", "ag"],
    ),
    (
        [TLazy("la", True), TLazy("lb", True), TNonLazy("c"), TLazy("ld", True)],
        {"reorder": "lazy_last"},
        ["c", "la", "lb", "ld"],
        ["c", "la", "lb", "ld"],
    ),
    (
        [TLazy("la", True), TLazy("lb", False), TNonLazy("c"), TLazy("ld", True)],
        {"reorder": "lazy_last"},
        ["lb", "c", "la", "ld"],
        ["c", "la", "lb", "ld"],
    ),
]


class TestTransformReordering(unittest.TestCase):
    @parameterized.expand(TRANSFORM_REORDERING_TEST_CASES)
    def test_transform_reordering_test_cases(self, transforms, options, lazy_enabled_expected, lazy_on_expected):
        with self.subTest("enable lazy"):
            c = ExecutionOptions()(transforms, lazy=None, options={"reorder": "lazy_last"})
            reordered = [transforms[i] for i in c["indices"]]
            actual = [t.tag for t in reordered]
            self.assertListEqual(actual, lazy_enabled_expected)

        with self.subTest("force lazy"):
            c = ExecutionOptions()(transforms, lazy=True, options={"reorder": "lazy_last"})
            reordered = [transforms[i] for i in c["indices"]]
            actual = [t.tag for t in reordered]
            self.assertListEqual(actual, lazy_on_expected)


if __name__ == "__main__":
    unittest.main()

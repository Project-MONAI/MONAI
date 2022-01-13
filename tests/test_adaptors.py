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

import itertools
import unittest

from monai.transforms.adaptors import FunctionSignature, adaptor, apply_alias, to_kwargs
from monai.utils.enums import CommonKeys

class TestAdaptors(unittest.TestCase):
    def test_function_signature(self):
        def foo(image, label=None, *a, **kw):
            pass

        f = FunctionSignature(foo)

    def test_single_in_single_out(self):
        def foo(image):
            return image * 2

        it = itertools.product(
            [CommonKeys.IMAGE, [CommonKeys.IMAGE]],
            [None, CommonKeys.IMAGE, [CommonKeys.IMAGE], {CommonKeys.IMAGE: CommonKeys.IMAGE}],
        )
        for i in it:
            d = {CommonKeys.IMAGE: 2}
            dres = adaptor(foo, i[0], i[1])(d)
            self.assertEqual(dres[CommonKeys.IMAGE], 4)

        d = {CommonKeys.IMAGE: 2}
        dres = adaptor(foo, CommonKeys.IMAGE)(d)
        self.assertEqual(dres[CommonKeys.IMAGE], 4)

        d = {CommonKeys.IMAGE: 2}
        dres = adaptor(foo, CommonKeys.IMAGE, CommonKeys.IMAGE)(d)
        self.assertEqual(dres[CommonKeys.IMAGE], 4)

        d = {CommonKeys.IMAGE: 2}
        dres = adaptor(foo, CommonKeys.IMAGE, {CommonKeys.IMAGE: CommonKeys.IMAGE})(d)
        self.assertEqual(dres[CommonKeys.IMAGE], 4)

        d = {"img": 2}
        dres = adaptor(foo, "img", {"img": CommonKeys.IMAGE})(d)
        self.assertEqual(dres["img"], 4)

        d = {"img": 2}
        dres = adaptor(foo, ["img"], {"img": CommonKeys.IMAGE})(d)
        self.assertEqual(dres["img"], 4)

    def test_multi_in_single_out(self):
        def foo(image, label):
            return image * label

        it = itertools.product(
            [CommonKeys.IMAGE, [CommonKeys.IMAGE]],
            [
                None,
                [CommonKeys.IMAGE, CommonKeys.LABEL],
                {CommonKeys.IMAGE: CommonKeys.IMAGE, CommonKeys.LABEL: CommonKeys.LABEL},
            ],
        )

        for i in it:
            d = {CommonKeys.IMAGE: 2, CommonKeys.LABEL: 3}
            dres = adaptor(foo, i[0], i[1])(d)
            self.assertEqual(dres[CommonKeys.IMAGE], 6)
            self.assertEqual(dres[CommonKeys.LABEL], 3)

        it = itertools.product(
            ["newimage", ["newimage"]],
            [
                None,
                [CommonKeys.IMAGE, CommonKeys.LABEL],
                {CommonKeys.IMAGE: CommonKeys.IMAGE, CommonKeys.LABEL: CommonKeys.LABEL},
            ],
        )

        for i in it:
            d = {CommonKeys.IMAGE: 2, CommonKeys.LABEL: 3}
            dres = adaptor(foo, i[0], i[1])(d)
            self.assertEqual(dres[CommonKeys.IMAGE], 2)
            self.assertEqual(dres[CommonKeys.LABEL], 3)
            self.assertEqual(dres["newimage"], 6)

        it = itertools.product(["img", ["img"]], [{"img": CommonKeys.IMAGE, "lbl": CommonKeys.LABEL}])

        for i in it:
            d = {"img": 2, "lbl": 3}
            dres = adaptor(foo, i[0], i[1])(d)
            self.assertEqual(dres["img"], 6)
            self.assertEqual(dres["lbl"], 3)

    def test_default_arg_single_out(self):
        def foo(a, b=2):
            return a * b

        d = {"a": 5}
        dres = adaptor(foo, "c")(d)
        self.assertEqual(dres["c"], 10)

        d = {"b": 5}
        with self.assertRaises(TypeError):
            dres = adaptor(foo, "c")(d)

    def test_multi_out(self):
        def foo(a, b):
            return a * b, a / b

        d = {"a": 3, "b": 4}
        dres = adaptor(foo, ["c", "d"])(d)
        self.assertEqual(dres["c"], 12)
        self.assertEqual(dres["d"], 3 / 4)

    def test_dict_out(self):
        def foo(a):
            return {"a": a * 2}

        d = {"a": 2}
        dres = adaptor(foo, {"a": "a"})(d)
        self.assertEqual(dres["a"], 4)

        d = {"b": 2}
        dres = adaptor(foo, {"a": "b"}, {"b": "a"})(d)
        self.assertEqual(dres["b"], 4)


class TestApplyAlias(unittest.TestCase):
    def test_apply_alias(self):
        def foo(d):
            d["x"] *= 2
            return d

        d = {"a": 1, "b": 3}
        result = apply_alias(foo, {"b": "x"})(d)
        self.assertDictEqual({"a": 1, "b": 6}, result)


class TestToKwargs(unittest.TestCase):
    def test_to_kwargs(self):
        def foo(**kwargs):
            results = {k: v * 2 for k, v in kwargs.items()}
            return results

        def compose_like(fn, data):
            data = fn(data)
            return data

        d = {"a": 1, "b": 2}

        actual = compose_like(to_kwargs(foo), d)
        self.assertDictEqual(actual, {"a": 2, "b": 4})

        with self.assertRaises(TypeError):
            actual = compose_like(foo, d)

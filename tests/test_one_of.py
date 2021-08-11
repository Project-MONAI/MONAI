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

from monai.transforms import InvertibleTransform, OneOf, Randomizable, Transform


class X(Transform):
    def __call__(self, x):
        return x


class Y(Transform):
    def __call__(self, x):
        return x


class A(Transform):
    def __call__(self, x):
        return x + 1


class B(Transform):
    def __call__(self, x):
        return x + 2


class C(Transform):
    def __call__(self, x):
        return x + 3


class Inv(InvertibleTransform):
    def __call__(self, x):
        return x + 1

    def inverse(self, x):
        return x - 1


class NonInv(Randomizable):
    def __call__(self, x):
        return x + self.R.uniform(-1, 1)


TESTS = [
    ((X(), Y(), X()), (1, 2, 1), (0.25, 0.5, 0.25)),
]


class TestOneOf(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_normalize_weights(self, transforms, input_weights, expected_weights):
        tr = OneOf(transforms, input_weights)
        self.assertTupleEqual(tr.weights, expected_weights)

    def test_no_weights_arg(self):
        p = OneOf((X(), Y(), X(), Y()))
        expected_weights = (0.25,) * 4
        self.assertTupleEqual(p.weights, expected_weights)

    def test_len_and_flatten(self):
        p1 = OneOf((X(), Y()), (1, 3))  # 0.25, 0.75
        p2 = OneOf((Y(), Y()), (2, 2))  # 0.5. 0.5
        p = OneOf((p1, p2, X()), (1, 2, 1))  # 0.25, 0.5, 0.25
        expected_order = (X, Y, Y, Y, X)
        expected_weights = (0.25 * 0.25, 0.25 * 0.75, 0.5 * 0.5, 0.5 * 0.5, 0.25)
        self.assertEqual(len(p), len(expected_order))
        self.assertTupleEqual(p.flatten().weights, expected_weights)

    def test_inverse(self):
        p = OneOf((OneOf((Inv(), NonInv())), Inv(), NonInv()))
        for _i in range(20):
            out = p(2.0)
            inverted = p.inverse(out)
            if p.index == 0 and p.transforms[0].index == 0 or p.index == 1:
                self.assertEqual(out, 3.0)
                self.assertEqual(inverted, 2.0)
            else:
                self.assertEqual(inverted, out)

    def test_one_of(self):
        p = OneOf((A(), B(), C()), (1, 2, 1))
        counts = [0] * 3
        for _i in range(10000):
            out = p(1.0)
            counts[int(out - 2)] += 1
        self.assertAlmostEqual(counts[0] / 10000, 0.25, delta=1.0)
        self.assertAlmostEqual(counts[1] / 10000, 0.50, delta=1.0)
        self.assertAlmostEqual(counts[2] / 10000, 0.25, delta=1.0)


if __name__ == "__main__":
    unittest.main()

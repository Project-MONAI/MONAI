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

from monai.transforms import OneOf, Transform


class X(Transform):
    def __call__(self, x):
        return x


class Y(Transform):
    def __call__(self, x):
        return x


TESTS = [
    ((X(), Y(), X()), (1, 2, 1), (0.25, 0.5, 0.25)),
]


class TestOneOf(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_one_of(self, transforms, input_weights, expected_weights):
        tr = OneOf(transforms, input_weights)
        self.assertTupleEqual(tr.weights, expected_weights)

    def test_len_and_flatten(self):
        p1 = OneOf((X(), Y()), (1, 3))  # 0.25, 0.75
        p2 = OneOf((Y(), Y()), (2, 2))  # 0.5. 0.5
        p = OneOf((p1, p2, X()), (1, 2, 1))  # 0.25, 0.5, 0.25
        expected_order = (X, Y, Y, Y, X)
        expected_weights = (0.25 * 0.25, 0.25 * 0.75, 0.5 * 0.5, 0.5 * 0.5, 0.25)
        self.assertEqual(len(p), len(expected_order))
        self.assertTupleEqual(p.flatten().weights, expected_weights)


if __name__ == "__main__":
    unittest.main()

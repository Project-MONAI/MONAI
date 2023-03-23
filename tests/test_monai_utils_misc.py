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

from parameterized import parameterized

from monai.utils.misc import to_tuple_of_dictionaries

TO_TUPLE_OF_DICTIONARIES_TEST_CASES = [
    ({}, tuple(), tuple()),
    ({}, ("x",), ({},)),
    ({}, ("x", "y"), ({}, {})),
    ({"a": 1}, tuple(), tuple()),
    ({"a": 1}, ("x",), ({"a": 1},)),
    ({"a": (1,)}, ("x",), ({"a": 1},)),
    ({"a": (1,)}, ("x", "y"), ValueError()),
    ({"a": 1}, ("x", "y"), ({"a": 1}, {"a": 1})),
    ({"a": (1, 2)}, tuple(), tuple()),
    ({"a": (1, 2)}, ("x", "y"), ({"a": 1}, {"a": 2})),
    ({"a": (1, 2, 3)}, ("x", "y"), ValueError()),
    ({"b": (2,), "a": 1}, tuple(), tuple()),
    ({"b": (2,), "a": 1}, ("x",), ({"b": 2, "a": 1},)),
    ({"b": (2,), "a": 1}, ("x", "y"), ValueError()),
    ({"b": (3, 2), "a": 1}, tuple(), tuple()),
    ({"b": (3, 2), "a": 1}, ("x",), ValueError()),
    ({"b": (3, 2), "a": 1}, ("x", "y"), ({"b": 3, "a": 1}, {"b": 2, "a": 1})),
]


class TestToTupleOfDictionaries(unittest.TestCase):
    @parameterized.expand(TO_TUPLE_OF_DICTIONARIES_TEST_CASES)
    def test_to_tuple_of_dictionaries(self, dictionary, keys, expected):
        self._test_to_tuple_of_dictionaries(dictionary, keys, expected)

    def _test_to_tuple_of_dictionaries(self, dictionary, keys, expected):
        if isinstance(expected, Exception):
            with self.assertRaises(type(expected)):
                to_tuple_of_dictionaries(dictionary, keys)
            print(type(expected))
        else:
            actual = to_tuple_of_dictionaries(dictionary, keys)
            print(actual, expected)
            self.assertTupleEqual(actual, expected)

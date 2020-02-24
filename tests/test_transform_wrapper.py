# Copyright 2020 MONAI Consortium
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

from monai.transforms.rand_rotation import ASYNC, OFF, SYNC, fn_map

TEST_CASES_1 = [
    [
        {'img': 1, 'seg': 2},
        {'transform': lambda x: x + 1},
        {'img': 2, 'seg': 3},
    ],
    [
        {'img': 1, 'seg': 2},
        {'transform': lambda x: x + 1, 'inplace': False, 'randomize': OFF},
        {'img': 2, 'seg': 3},
    ],
    [
        {'img': 1, 'seg': 2},
        {'transform': lambda x: x + 1, 'map_key': 'img', 'inplace': False, 'randomize': OFF},
        {'img': 2, 'seg': 2},
    ],
    [
        {'img': 1, 'seg': 2},
        {'transform': lambda x: x + 1, 'map_key': ['img'], 'inplace': False, 'randomize': OFF},
        {'img': 2, 'seg': 2},
    ],
]

TEST_VALUEERROR_CASES = [
    [
        {'img': 1, 'seg': 2},
        {'transform': lambda x: x + 1, 'randomize': SYNC},
        '.*not randomizable.*',
    ],
    [
        {'img': 1, 'seg': 2},
        {'transform': lambda x: x + 1, 'randomize': ASYNC},
        '.*not randomizable.*',
    ],
    [
        1,
        {'transform': lambda x: x + 1},
        '.*dictionary.*',
    ],
]
TEST_KEYERROR_CASES = [
    [
        {'img': 1, 'seg': 2},
        {'transform': lambda x: x + 1, 'map_key': 'foo'},
        '.*foo.*',
    ],
]


class TestTransformWrapper(unittest.TestCase):
    # fn_map(transform, map_key, common_key, inplace, randomize)

    @parameterized.expand(TEST_CASES_1)
    def test_wrapped(self, data, test_args, expected):
        output = fn_map(**test_args)(data)
        self.assertDictEqual(expected, output)

    @parameterized.expand(TEST_VALUEERROR_CASES)
    def test_wrong_args(self, data, test_args, expected_msg):
        with self.assertRaisesRegex(ValueError, expected_msg):
            fn_map(**test_args)(data)

    @parameterized.expand(TEST_KEYERROR_CASES)
    def test_wrong_keys(self, data, test_args, expected_msg):
        with self.assertRaisesRegex((KeyError, TypeError, AssertionError), expected_msg):
            fn_map(**test_args)(data)


if __name__ == '__main__':
    unittest.main()

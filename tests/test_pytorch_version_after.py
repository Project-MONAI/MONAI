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

import unittest

from parameterized import parameterized

from monai.utils import pytorch_after

TEST_CASES = (
    (1, 5, 9, "1.6.0"),
    (1, 6, 0, "1.6.0"),
    (1, 6, 1, "1.6.0", False),
    (1, 7, 0, "1.6.0", False),
    (2, 6, 0, "1.6.0", False),
    (0, 6, 0, "1.6.0a0+3fd9dcf"),
    (1, 5, 9, "1.6.0a0+3fd9dcf"),
    (1, 6, 0, "1.6.0a0+3fd9dcf", False),
    (1, 6, 1, "1.6.0a0+3fd9dcf", False),
    (2, 6, 0, "1.6.0a0+3fd9dcf", False),
    (1, 6, 0, "1.6.0-rc0+3fd9dcf", False),  # defaults to prerelease
    (1, 6, 0, "1.6.0rc0", False),
    (1, 6, 0, "1.6", True),
    (1, 6, 0, "1", False),
    (1, 6, 0, "1.6.0+cpu", True),
    (1, 6, 1, "1.6.0+cpu", False),
)


class TestPytorchVersionCompare(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_compare(self, a, b, p, current, expected=True):
        """Test pytorch_after with a and b"""
        self.assertEqual(pytorch_after(a, b, p, current), expected)


if __name__ == "__main__":
    unittest.main()

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
from tests.utils import TEST_NDARRAYS, assert_allclose

from parameterized import parameterized

from monai.generation.maisi.utils import morphological_ops

TESTS = []
for p in TEST_NDARRAYS:
    mask = torch.zeros(1,1,5,5,5)
    filter_size = 3
    TESTS.append(
        [
            {"mask": p(mask), "filter_size": filter_size},
            p(torch.zeros(1,1,5,5,5)),
        ]
    )

class TestMorph(unittest.TestCase):

    @parameterized.expand(TESTS)
    def test_value(self, input_data, expected_result):
        result1 = morphological_ops.erode(input_data["mask"],input_data["filter_size"])
        assert_allclose(result2, expected_box, type_test=True, device_test=True, atol=0.0)

if __name__ == "__main__":
    unittest.main()

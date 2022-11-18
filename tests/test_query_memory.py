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

from tests.utils import query_memory


class TestQueryMemory(unittest.TestCase):
    def test_output_str(self):
        self.assertTrue(isinstance(query_memory(2), str))
        all_device = query_memory(-1)
        self.assertTrue(isinstance(all_device, str))
        self.assertEqual(query_memory("test"), "")


if __name__ == "__main__":
    unittest.main()

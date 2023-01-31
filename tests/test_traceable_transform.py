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

from monai.data import MetaTensor
from monai.transforms.inverse import TraceableTransform


class _TraceTest(TraceableTransform):
    def __call__(self, data):
        self.push_transform(data, "image")
        return data

    def pop(self, data):
        self.pop_transform(data, "image")
        return data


class TestTraceable(unittest.TestCase):
    def test_default(self):
        expected_key = "_transforms"
        a = _TraceTest()
        for x in a.transform_keys():
            self.assertTrue(x in a.get_transform_info())
        self.assertEqual(a.trace_key(), expected_key)

        data = {"image": "test"}
        data = a(data)  # adds to the stack
        self.assertEqual(data["image"], "test")

        data = {"image": MetaTensor(1.0)}
        data = a(data)  # adds to the stack
        self.assertEqual(data["image"].applied_operations[0]["class"], "_TraceTest")


if __name__ == "__main__":
    unittest.main()

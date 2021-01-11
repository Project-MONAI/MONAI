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

from monai.transforms.utility.dictionary import Identityd
from tests.utils import NumpyImageTestCase2D


class TestIdentityd(NumpyImageTestCase2D):
    def test_identityd(self):
        img = self.imt
        data = dict()
        data["img"] = img
        identity = Identityd(keys=data.keys())
        self.assertEqual(data, identity(data))


if __name__ == "__main__":
    unittest.main()

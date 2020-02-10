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

from monai.data.transforms.adaptors import adaptor


class TestAdaptors(unittest.TestCase):

    def test_single_in_single_out(self):
        def foo(image):
            return image * 2

        d = {'image': 2}
        dres = adaptor(foo, 'image')(d)
        self.assertEqual(dres['image'], 4)

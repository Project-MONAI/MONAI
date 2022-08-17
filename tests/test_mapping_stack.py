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

from monai.transforms.utils import TransformBackends

from monai.utils.mapping_stack import MappingStack, MatrixFactory


class MappingStackTest(unittest.TestCase):

    def test_scale_then_translate(self):

        f = MatrixFactory(3, TransformBackends.NUMPY)
        m_scale = f.scale((2, 2, 2))
        m_trans = f.translate((20, 20, 0))
        ms = MappingStack(f)
        ms.push(m_scale)
        ms.push(m_trans)

        print(ms.transform()._matrix)

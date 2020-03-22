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

import numpy as np

from monai.transforms.composables import Orientationd


class TestOrientationdCase(unittest.TestCase):

    def test_orntd(self):
        data = {'seg': np.ones((2, 1, 2, 3)), 'affine': np.eye(4)}
        ornt = Orientationd(keys='seg', affine_key='affine', axcodes='RAS')
        res = ornt(data)
        np.testing.assert_allclose(res['seg'].shape, (2, 1, 2, 3))
        self.assertEqual(res['orientation']['original_ornt'], ('R', 'A', 'S'))
        self.assertEqual(res['orientation']['current_ornt'], 'RAS')

    def test_orntd_3d(self):
        data = {'seg': np.ones((2, 1, 2, 3)), 'img': np.ones((2, 1, 2, 3)), 'affine': np.eye(4)}
        ornt = Orientationd(keys=('img', 'seg'), affine_key='affine', axcodes='PLI')
        res = ornt(data)
        np.testing.assert_allclose(res['img'].shape, (2, 2, 1, 3))
        self.assertEqual(res['orientation']['original_ornt'], ('R', 'A', 'S'))
        self.assertEqual(res['orientation']['current_ornt'], 'PLI')

    def test_orntd_2d(self):
        data = {'seg': np.ones((2, 1, 3)), 'img': np.ones((2, 1, 3)), 'affine': np.eye(4)}
        ornt = Orientationd(keys=('img', 'seg'), affine_key='affine', axcodes='PLI')
        res = ornt(data)
        np.testing.assert_allclose(res['img'].shape, (2, 3, 1))
        self.assertEqual(res['orientation']['original_ornt'], ('R', 'A'))
        self.assertEqual(res['orientation']['current_ornt'], 'PL')

    def test_orntd_1d(self):
        data = {'seg': np.ones((2, 3)), 'img': np.ones((2, 3)), 'affine': np.eye(4)}
        ornt = Orientationd(keys=('img', 'seg'), affine_key='affine', axcodes='L')
        res = ornt(data)
        np.testing.assert_allclose(res['img'].shape, (2, 3))
        self.assertEqual(res['orientation']['original_ornt'], ('R',))
        self.assertEqual(res['orientation']['current_ornt'], 'L')


if __name__ == '__main__':
    unittest.main()

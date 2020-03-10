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

from monai.transforms.composables import Spacingd


class TestSpacingDCase(unittest.TestCase):

    def test_spacingd_3d(self):
        data = {'image': np.ones((2, 10, 15, 20)), 'affine': np.eye(4)}
        spacing = Spacingd(keys='image', affine_key='affine', pixdim=(1, 2, 1.4))
        res = spacing(data)
        np.testing.assert_allclose(res['image'].shape, (2, 10, 8, 14))
        np.testing.assert_allclose(res['spacing']['current_pixdim'], (1, 2, 1.4))
        np.testing.assert_allclose(res['spacing']['original_pixdim'], (1, 1, 1))

    def test_spacingd_2d(self):
        data = {'image': np.ones((2, 10, 20)), 'affine': np.eye(4)}
        spacing = Spacingd(keys='image', affine_key='affine', pixdim=(1, 2, 1.4))
        res = spacing(data)
        np.testing.assert_allclose(res['image'].shape, (2, 10, 10))
        np.testing.assert_allclose(res['spacing']['current_pixdim'], (1, 2))
        np.testing.assert_allclose(res['spacing']['original_pixdim'], (1, 1))

    def test_spacingd_1d(self):
        data = {'image': np.ones((2, 10)), 'affine': np.eye(4)}
        spacing = Spacingd(keys='image', affine_key='affine', pixdim=(0.2,))
        res = spacing(data)
        np.testing.assert_allclose(res['image'].shape, (2, 50))
        np.testing.assert_allclose(res['spacing']['current_pixdim'], (0.2,))
        np.testing.assert_allclose(res['spacing']['original_pixdim'], (1,))

    def test_interp_all(self):
        data = {'image': np.ones((2, 10)), 'seg': np.ones((2, 10)), 'affine': np.eye(4)}
        spacing = Spacingd(keys=('image', 'seg'), affine_key='affine', interp_order=0, pixdim=(0.2,))
        res = spacing(data)
        np.testing.assert_allclose(res['image'].shape, (2, 50))
        np.testing.assert_allclose(res['spacing']['current_pixdim'], (0.2,))
        np.testing.assert_allclose(res['spacing']['original_pixdim'], (1,))

    def test_interp_sep(self):
        data = {'image': np.ones((2, 10)), 'seg': np.ones((2, 10)), 'affine': np.eye(4)}
        spacing = Spacingd(keys=('image', 'seg'), affine_key='affine', interp_order=(2, 0), pixdim=(0.2,))
        res = spacing(data)
        np.testing.assert_allclose(res['image'].shape, (2, 50))
        np.testing.assert_allclose(res['spacing']['current_pixdim'], (0.2,))
        np.testing.assert_allclose(res['spacing']['original_pixdim'], (1,))


if __name__ == '__main__':
    unittest.main()

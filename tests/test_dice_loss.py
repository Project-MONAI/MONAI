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

import torch
from parameterized import parameterized

from monai.losses import DiceLoss

TEST_CASE_1 = [  # shape: (1, 1, 2, 2), (1, 1, 2, 2)
    {
        'include_background': True,
        'do_sigmoid': True,
    },
    {
        'pred': torch.tensor([[[[1., -1.], [-1., 1.]]]]),
        'ground': torch.tensor([[[[1., 0.], [1., 1.]]]]),
        'smooth': 1e-6,
    },
    0.307576,
]

TEST_CASE_2 = [  # shape: (2, 1, 2, 2), (2, 1, 2, 2)
    {
        'include_background': True,
        'do_sigmoid': True,
    },
    {
        'pred': torch.tensor([[[[1., -1.], [-1., 1.]]], [[[1., -1.], [-1., 1.]]]]),
        'ground': torch.tensor([[[[1., 1.], [1., 1.]]], [[[1., 0.], [1., 0.]]]]),
        'smooth': 1e-4,
    },
    0.416657,
]

TEST_CASE_3 = [  # shape: (2, 2, 3), (2, 1, 3)
    {
        'include_background': False,
        'to_onehot_y': True,
    },
    {
        'pred': torch.tensor([[[1., 1., 0.], [0., 0., 1.]], [[1., 0., 1.], [0., 1., 0.]]]),
        'ground': torch.tensor([[[0., 0., 1.]], [[0., 1., 0.]]]),
        'smooth': 0.0,
    },
    0.0,
]

TEST_CASE_4 = [  # shape: (2, 2, 3), (2, 1, 3)
    {
        'include_background': True,
        'to_onehot_y': True,
        'do_sigmoid': True,
    },
    {
        'pred': torch.tensor([[[-1., 0., 1.], [1., 0., -1.]], [[0., 0., 0.], [0., 0., 0.]]]),
        'ground': torch.tensor([[[1., 0., 0.]], [[1., 1., 0.]]]),
        'smooth': 1e-4,
    },
    0.435050,
]

TEST_CASE_5 = [  # shape: (2, 2, 3), (2, 1, 3)
    {
        'include_background': True,
        'to_onehot_y': True,
        'do_softmax': True,
    },
    {
        'pred': torch.tensor([[[-1., 0., 1.], [1., 0., -1.]], [[0., 0., 0.], [0., 0., 0.]]]),
        'ground': torch.tensor([[[1., 0., 0.]], [[1., 1., 0.]]]),
        'smooth': 1e-4,
    },
    0.383713,
]

TEST_CASE_6 = [  # shape: (1, 1, 2, 2), (1, 1, 2, 2)
    {
        'include_background': True,
        'do_sigmoid': True,
    },
    {
        'pred': torch.tensor([[[[1., -1.], [-1., 1.]]]]),
        'ground': torch.tensor([[[[1., 0.], [1., 1.]]]]),
        'smooth': 1e-6,
    },
    0.307576,
]

TEST_CASE_7 = [  # shape: (1, 1, 2, 2), (1, 1, 2, 2)
    {
        'include_background': True,
        'do_sigmoid': True,
        'squared_pred': True,
    },
    {
        'pred': torch.tensor([[[[1., -1.], [-1., 1.]]]]),
        'ground': torch.tensor([[[[1., 0.], [1., 1.]]]]),
        'smooth': 1e-5,
    },
    0.178337,
]

TEST_CASE_8 = [  # shape: (1, 1, 2, 2), (1, 1, 2, 2)
    {
        'include_background': True,
        'do_sigmoid': True,
        'jaccard': True,
    },
    {
        'pred': torch.tensor([[[[1., -1.], [-1., 1.]]]]),
        'ground': torch.tensor([[[[1., 0.], [1., 1.]]]]),
        'smooth': 1e-5,
    },
    -0.059094,
]


class TestDiceLoss(unittest.TestCase):

    @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_3, TEST_CASE_4,
                           TEST_CASE_5, TEST_CASE_6, TEST_CASE_7, TEST_CASE_8])
    def test_shape(self, input_param, input_data, expected_val):
        result = DiceLoss(**input_param).forward(**input_data)
        self.assertAlmostEqual(result.item(), expected_val, places=5)


if __name__ == '__main__':
    unittest.main()

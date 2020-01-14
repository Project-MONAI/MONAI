import unittest

import torch
from parameterized import parameterized

from monai.networks.losses.dice import DiceLoss


class TestDiceLoss(unittest.TestCase):

    @parameterized.expand([
        [
            {
                'include_background': False,
            },
            {
                'pred': torch.tensor([[[[1., -1.], [-1., 1.]]]]),
                'ground': torch.tensor([[[[1., 0.], [1., 1.]]]]),
                'smooth': 1e-6,
            },
            0.307576,
        ],
        [
            {
                'include_background': True,
            },
            {
                'pred': torch.tensor([[[[1., -1.], [-1., 1.]]], [[[1., -1.], [-1., 1.]]]]),
                'ground': torch.tensor([[[[1., 1.], [1., 1.]]], [[[1., 0.], [1., 0.]]]]),
                'smooth': 1e-4,
            },
            0.416636,
        ],
    ])
    def test_shape(self, input_param, input_data, expected_val):
        result = DiceLoss(**input_param).forward(**input_data)
        self.assertAlmostEqual(result.item(), expected_val, places=5)


if __name__ == '__main__':
    unittest.main()

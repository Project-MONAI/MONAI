import unittest

import torch
from parameterized import parameterized

from monai.engines.prepare_batch import PrepareBatchShuffle
from monai.utils import set_determinism

set_determinism(0)

TEST_CASE_1 = [
    {"image": torch.Tensor([1.0, 2.0, 3.0]), "label": torch.Tensor([11.0, 22.0, 33.0])},
    (torch.Tensor([3.0, 1.0, 2.0]), torch.Tensor([33.0, 11.0, 22.0])),
]

TEST_CASE_2 = [
    {
        "image": torch.stack([1.0 * torch.ones((2, 2)), 2.0 * torch.ones((2, 2)), 3.0 * torch.ones((2, 2))]),
        "label": torch.Tensor([11.0, 22.0, 33.0]),
    },
    (
        torch.stack([3.0 * torch.ones((2, 2)), 2.0 * torch.ones((2, 2)), 1.0 * torch.ones((2, 2))]),
        torch.Tensor([33.0, 22.0, 11.0]),
    ),
]

# Image only
TEST_CASE_3 = [
    {"image": torch.stack([1.0 * torch.ones((2, 2)), 2.0 * torch.ones((2, 2)), 3.0 * torch.ones((2, 2))])},
    (torch.stack([2.0 * torch.ones((2, 2)), 3.0 * torch.ones((2, 2)), 1.0 * torch.ones((2, 2))]), None),
]


class TestPrepareBatch(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_3])
    def test_prepare_batch_shuffle(self, data, expected_output):
        prepare_batch = PrepareBatchShuffle()
        output = prepare_batch(batchdata=data)
        self.assertTrue(torch.equal(output[0], expected_output[0]))
        if expected_output[1] is None:
            self.assertIsNone(output[1])
        else:
            self.assertTrue(torch.equal(output[1], expected_output[1]))


if __name__ == "__main__":
    unittest.main()

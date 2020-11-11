import unittest

import torch
from parameterized import parameterized

from monai.networks.nets import FCNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

TEST_CASE_0 = [0]
TEST_CASE_1 = [0.15]

CASES = [TEST_CASE_0, TEST_CASE_1]


class TestFCnet(unittest.TestCase):
    def setUp(self):
        self.batch_size = 10
        self.inSize = 10
        self.arrShape = (self.batch_size, self.inSize)
        self.outSize = 3
        self.channels = [8, 16]
        self.arr = torch.randn(self.arrShape, dtype=torch.float32).to(device)

    @parameterized.expand(CASES)
    def test_shape(self, dropout):
        net = FCNet(self.inSize, self.outSize, self.channels, dropout).to(device)
        out = net(self.arr)
        self.assertEqual(out.shape, (self.batch_size, self.outSize))


if __name__ == "__main__":
    unittest.main()

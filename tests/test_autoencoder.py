import unittest

from monai.networks.nets.autoencoder import AutoEncoder


class TestAutoEncoder(unittest.TestCase):
    def test_simple1(self):
        net = AutoEncoder(2, 1, 1, [4, 8, 16], [2, 2, 2])
        out = net(self.imT)
        self.assertEqual(out[0].shape, self.imT.shape)

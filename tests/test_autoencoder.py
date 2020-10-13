import unittest

import torch

from monai.data import create_test_image_2d
from monai.networks.nets.autoencoder import AutoEncoder


class TestAutoEncoder(unittest.TestCase):
    def test_simple1(self):

        im_shape = (128, 128)
        input_channels = 1
        output_channels = 4
        num_classes = 3
        im, _ = create_test_image_2d(im_shape[0], im_shape[1], 4, 20, 0, num_classes)
        imT = torch.tensor(im[None, None])

        net = AutoEncoder(2, 1, 1, [4, 8, 16], [2, 2, 2])
        out = net(imT)
        self.assertEqual(out.shape, imT.shape)


if __name__ == "__main__":
    unittest.main()

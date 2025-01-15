import unittest
import torch
from monai.networks.utils import pixelunshuffle, pixelshuffle

class TestPixelUnshuffle(unittest.TestCase):

    def test_2d_basic(self):
        x = torch.randn(2, 4, 16, 16)
        out = pixelunshuffle(x, spatial_dims=2, scale_factor=2)
        self.assertEqual(out.shape, (2, 16, 8, 8))

    def test_3d_basic(self):
        x = torch.randn(2, 4, 16, 16, 16) 
        out = pixelunshuffle(x, spatial_dims=3, scale_factor=2)
        self.assertEqual(out.shape, (2, 32, 8, 8, 8))

    def test_inverse_pixelshuffle(self):
        x = torch.randn(2, 4, 16, 16)
        shuffled = pixelshuffle(x, spatial_dims=2, scale_factor=2) 
        unshuffled = pixelunshuffle(shuffled, spatial_dims=2, scale_factor=2)
        torch.testing.assert_close(x, unshuffled)

    def test_compare_torch_pixel_unshuffle(self):
        x = torch.randn(2, 4, 16, 16)
        monai_out = pixelunshuffle(x, spatial_dims=2, scale_factor=2)
        torch_out = torch.pixel_unshuffle(x, downscale_factor=2)
        torch.testing.assert_close(monai_out, torch_out)

    def test_invalid_scale(self):
        x = torch.randn(2, 4, 15, 15)
        with self.assertRaises(RuntimeError):
            pixelunshuffle(x, spatial_dims=2, scale_factor=2)

if __name__ == "__main__":
    unittest.main()

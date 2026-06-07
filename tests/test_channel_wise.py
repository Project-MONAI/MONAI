import unittest

import numpy as np
import torch

from monai.transforms import ChannelWise, RandChannelWise, RandGaussianNoise, ScaleIntensity
from monai.utils import set_determinism


EXPECTED_SCALED = np.array([[0.0, 0.3333333], [0.6666667, 1.0]])


class TestChannelWise(unittest.TestCase):
    def test_channel_wise_deterministic(self):
        data = np.array([[[1.0, 2.0], [3.0, 4.0]], [[10.0, 20.0], [30.0, 40.0]]])

        transform = ChannelWise(transform=ScaleIntensity())
        out = transform(data)

        np.testing.assert_allclose(out[0], EXPECTED_SCALED, atol=1e-5)
        np.testing.assert_allclose(out[1], EXPECTED_SCALED, atol=1e-5)
        self.assertEqual(out.shape, data.shape)

        torch_data = torch.as_tensor(data)
        torch_out = transform(torch_data)

        torch_expected = torch.as_tensor(EXPECTED_SCALED, dtype=torch_out.dtype)
        self.assertTrue(torch.allclose(torch_out[0], torch_expected, atol=1e-5))
        self.assertTrue(torch.allclose(torch_out[1], torch_expected, atol=1e-5))
        self.assertEqual(torch_out.shape, torch_data.shape)

    def test_rand_channel_wise(self):
        try:
            set_determinism(seed=0)

            data = np.zeros((3, 4, 4))
            transform = RandChannelWise(transform=RandGaussianNoise(prob=1.0, std=1.0))
            out = transform(data)

            self.assertFalse(np.allclose(out[0], out[1]))
            self.assertFalse(np.allclose(out[1], out[2]))
            self.assertFalse(np.allclose(out[0], out[2]))
            self.assertEqual(out.shape, data.shape)

            torch_data = torch.zeros((3, 4, 4))
            torch_transform = RandChannelWise(transform=RandGaussianNoise(prob=1.0, std=1.0))
            torch_out = torch_transform(torch_data)

            self.assertFalse(torch.allclose(torch_out[0], torch_out[1]))
            self.assertFalse(torch.allclose(torch_out[1], torch_out[2]))
            self.assertFalse(torch.allclose(torch_out[0], torch_out[2]))
            self.assertEqual(torch_out.shape, torch_data.shape)
        finally:
            set_determinism(None)

    def test_prob_zero(self):
        data = np.zeros((2, 2, 2))
        transform = RandChannelWise(transform=RandGaussianNoise(prob=1.0, std=1.0), prob=0.0)
        out = transform(data)
        np.testing.assert_allclose(out, data)

        torch_data = torch.zeros((2, 2, 2))
        torch_out = transform(torch_data)
        self.assertTrue(torch.allclose(torch_out, torch_data))

    def test_squeezed_channel_result(self):
        data = np.arange(8.0).reshape(2, 2, 2)
        transform = ChannelWise(transform=lambda img: img[0])
        out = transform(data)
        np.testing.assert_allclose(out, data)
        self.assertEqual(out.shape, data.shape)

        torch_data = torch.as_tensor(data)
        torch_out = transform(torch_data)
        self.assertTrue(torch.allclose(torch_out, torch_data))
        self.assertEqual(torch_out.shape, torch_data.shape)

    def test_invalid_channel_result_shape(self):
        transform = ChannelWise(transform=lambda img: img[:0])

        with self.assertRaises(ValueError):
            transform(np.zeros((2, 2, 2)))

        with self.assertRaises(ValueError):
            transform(torch.zeros((2, 2, 2)))


if __name__ == "__main__":
    unittest.main()

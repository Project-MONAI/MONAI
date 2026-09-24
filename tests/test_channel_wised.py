import unittest

import numpy as np
import torch

from monai.transforms import ChannelWised, RandChannelWised, RandGaussianNoise, ScaleIntensity
from monai.utils import set_determinism


EXPECTED_SCALED = np.array([[0.0, 0.3333333], [0.6666667, 1.0]])


class TestChannelWised(unittest.TestCase):
    def test_channel_wise_deterministic(self):
        data = {"image": np.array([[[1.0, 2.0], [3.0, 4.0]], [[10.0, 20.0], [30.0, 40.0]]])}

        transform = ChannelWised(keys=["image"], transform=ScaleIntensity())
        out = transform(data)

        np.testing.assert_allclose(out["image"][0], EXPECTED_SCALED, atol=1e-5)
        np.testing.assert_allclose(out["image"][1], EXPECTED_SCALED, atol=1e-5)
        self.assertEqual(out["image"].shape, data["image"].shape)

        torch_data = {"image": torch.as_tensor(data["image"])}
        torch_out = transform(torch_data)

        torch_expected = torch.as_tensor(EXPECTED_SCALED, dtype=torch_out["image"].dtype)
        self.assertTrue(torch.allclose(torch_out["image"][0], torch_expected, atol=1e-5))
        self.assertTrue(torch.allclose(torch_out["image"][1], torch_expected, atol=1e-5))
        self.assertEqual(torch_out["image"].shape, torch_data["image"].shape)

    def test_rand_channel_wise(self):
        try:
            set_determinism(seed=0)

            data = {"image": np.zeros((3, 4, 4))}
            transform = RandChannelWised(keys=["image"], transform=RandGaussianNoise(prob=1.0, std=1.0))
            out = transform(data)

            self.assertFalse(np.allclose(out["image"][0], out["image"][1]))
            self.assertFalse(np.allclose(out["image"][1], out["image"][2]))
            self.assertFalse(np.allclose(out["image"][0], out["image"][2]))
            self.assertEqual(out["image"].shape, data["image"].shape)

            torch_data = {"image": torch.zeros((3, 4, 4))}
            torch_transform = RandChannelWised(keys=["image"], transform=RandGaussianNoise(prob=1.0, std=1.0))
            torch_out = torch_transform(torch_data)

            self.assertFalse(torch.allclose(torch_out["image"][0], torch_out["image"][1]))
            self.assertFalse(torch.allclose(torch_out["image"][1], torch_out["image"][2]))
            self.assertFalse(torch.allclose(torch_out["image"][0], torch_out["image"][2]))
            self.assertEqual(torch_out["image"].shape, torch_data["image"].shape)
        finally:
            set_determinism(None)

    def test_prob_zero(self):
        data = {"image": np.zeros((2, 2, 2))}
        transform = RandChannelWised(keys=["image"], transform=RandGaussianNoise(prob=1.0, std=1.0), prob=0.0)
        out = transform(data)
        np.testing.assert_allclose(out["image"], data["image"])

        torch_data = {"image": torch.zeros((2, 2, 2))}
        torch_out = transform(torch_data)
        self.assertTrue(torch.allclose(torch_out["image"], torch_data["image"]))


if __name__ == "__main__":
    unittest.main()

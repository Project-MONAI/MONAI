import unittest

import numpy as np
import torch

from monai.transforms import ChannelWised, RandChannelWised, RandGaussianNoise, ScaleIntensity
from monai.utils import set_determinism


class TestChannelWised(unittest.TestCase):
    def test_channel_wise_deterministic(self):
        # Test applying a deterministic transform channel-wise
        data = {"image": np.array([[[1.0, 2.0], [3.0, 4.0]], [[10.0, 20.0], [30.0, 40.0]]])}  # shape (2, 2, 2)
        
        # ScaleIntensity applies to the whole input array independently
        transform = ChannelWised(keys=["image"], transform=ScaleIntensity())
        out = transform(data)
        
        # Channel 0 scaled
        np.testing.assert_allclose(out["image"][0], np.array([[0.0, 0.3333333], [0.6666667, 1.0]]), atol=1e-5)
        # Channel 1 scaled
        np.testing.assert_allclose(out["image"][1], np.array([[0.0, 0.3333333], [0.6666667, 1.0]]), atol=1e-5)
        self.assertEqual(out["image"].shape, data["image"].shape)

    def test_rand_channel_wise(self):
        # Test applying a randomized transform channel-wise
        data = {"image": np.zeros((3, 4, 4))}
        
        set_determinism(seed=0)
        # Apply random noise with high standard deviation to see the difference
        transform = RandChannelWised(keys=["image"], transform=RandGaussianNoise(prob=1.0, std=1.0))
        out = transform(data)
        
        # All channels should have different noise values
        self.assertFalse(np.allclose(out["image"][0], out["image"][1]))
        self.assertFalse(np.allclose(out["image"][1], out["image"][2]))
        self.assertFalse(np.allclose(out["image"][0], out["image"][2]))
        
        # Output shape should be exactly the same
        self.assertEqual(out["image"].shape, data["image"].shape)

    def test_prob_zero(self):
        # Test when RandChannelWised prob is 0.0
        data = {"image": np.zeros((2, 2, 2))}
        transform = RandChannelWised(keys=["image"], transform=RandGaussianNoise(prob=1.0, std=1.0), prob=0.0)
        out = transform(data)
        np.testing.assert_allclose(out["image"], data["image"])

if __name__ == "__main__":
    unittest.main()

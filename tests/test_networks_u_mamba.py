import unittest
import torch
from monai.networks.nets import UMamba

class TestUMamba(unittest.TestCase):
    def test_forward_shape(self):
        # Set up input dimensions and model
        input_tensor = torch.randn(2, 1, 64, 64)  # (batch_size, channels, H, W)
        model = UMamba(in_channels=1, out_channels=2)  # example args

        # Forward pass
        output = model(input_tensor)

        # Assert output shape matches expectation
        self.assertEqual(output.shape, (2, 2, 64, 64))  # adjust if necessary

    def test_script(self):
        # Test JIT scripting if supported
        model = UMamba(in_channels=1, out_channels=2)
        scripted = torch.jit.script(model)
        x = torch.randn(1, 1, 64, 64)
        out = scripted(x)
        self.assertEqual(out.shape, (1, 2, 64, 64))

if __name__ == "__main__":
    unittest.main()

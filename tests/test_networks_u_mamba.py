import unittest
import torch
from monai.networks.nets import UMambaUNet

class TestUMamba(unittest.TestCase):
    def test_forward_shape(self):
        # Set up input dimensions and model
        input_tensor = torch.randn(2, 1, 16, 64, 64)
        model = UMambaUNet(in_channels=1, out_channels=2)
        output = model(input_tensor)
        self.assertEqual(output.shape, (2, 2, 16, 64, 64))

    def test_script(self):
        # Test JIT scripting if supported
        model = UMambaUNet(in_channels=1, out_channels=2)
        scripted = torch.jit.script(model)
        x = torch.randn(1, 1, 64, 64)
        out = scripted(x)
        self.assertEqual(out.shape, (1, 2, 64, 64))

if __name__ == "__main__":
    unittest.main()

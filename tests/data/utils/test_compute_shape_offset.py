import torch
import unittest
import numpy as np
from monai.data.utils import compute_shape_offset

class TestComputeShapeOffsetRegression(unittest.TestCase):
    def test_pytorch_size_input(self):
        # 1 Create a PyTorch Size object (which triggered the original bug)
        spatial_shape = torch.Size([10, 10, 10])
        in_affine = np.eye(4)
        out_affine = np.eye(4)

        # 2 Feed it into the function
        shape, offset = compute_shape_offset(spatial_shape, in_affine, out_affine)

        # 3 Prove it successfully processed the shape by checking its length
        self.assertEqual(len(shape), 3)

if __name__ == "__main__":
    unittest.main()

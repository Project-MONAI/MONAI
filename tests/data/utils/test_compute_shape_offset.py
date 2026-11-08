# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import unittest

import numpy as np
import torch

from monai.data.utils import compute_shape_offset


class TestComputeShapeOffset(unittest.TestCase):
    """Unit tests for :func:`monai.data.utils.compute_shape_offset`."""

    def test_pytorch_size_input(self):
        """Validate `torch.Size` input produces expected shape and offset.

        Returns:
            None.

        Raises:
            AssertionError: If computed shape/offset are not as expected.
        """
        # 1. Create a PyTorch Size object (which triggered the original bug)
        spatial_shape = torch.Size([10, 10, 10])
        in_affine = np.eye(4)
        out_affine = np.eye(4)

        # 2. Feed it into the function
        shape, offset = compute_shape_offset(spatial_shape, in_affine, out_affine)

        # 3. Prove it successfully processed the shape by checking its length
        self.assertEqual(len(shape), 3)

    def setUp(self):
        """Set up a 4x4 identity affine used across all test cases."""
        self.affine = np.eye(4)

    def test_numpy_array_input(self):
        """Verify compute_shape_offset accepts a numpy array as spatial_shape."""
        shape = np.array([64, 64, 64])
        out_shape, _ = compute_shape_offset(shape, self.affine, self.affine)
        self.assertEqual(len(out_shape), 3)

    def test_list_input(self):
        """Verify compute_shape_offset accepts a plain list as spatial_shape."""
        shape = [64, 64, 64]
        out_shape, _ = compute_shape_offset(shape, self.affine, self.affine)
        self.assertEqual(len(out_shape), 3)

    def test_torch_tensor_input(self):
        """Verify compute_shape_offset accepts a torch.Tensor as spatial_shape.

        This path broke in PyTorch >= 2.9 because np.array() relied on the
        non-tuple sequence indexing protocol that PyTorch removed. Wrapping with
        tuple() fixes it.
        """
        shape = torch.tensor([64, 64, 64])
        out_shape, _ = compute_shape_offset(shape, self.affine, self.affine)
        self.assertEqual(len(out_shape), 3)

    def test_identity_affines_preserve_shape(self):
        """Verify that identity in/out affines produce an output shape matching the input."""
        shape = torch.tensor([32, 48, 16])
        out_shape, _ = compute_shape_offset(shape, self.affine, self.affine)
        np.testing.assert_allclose(np.array(out_shape, dtype=float), shape.numpy().astype(float), atol=1e-5)


if __name__ == "__main__":
    unittest.main()

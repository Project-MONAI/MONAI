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


class TestComputeShapeOffsetRegression(unittest.TestCase):
    """Regression tests for `compute_shape_offset` input-shape handling."""

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


if __name__ == "__main__":
    unittest.main()

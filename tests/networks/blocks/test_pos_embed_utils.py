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

from monai.networks.blocks.pos_embed_utils import build_sincos_position_embedding


class TestBuildSincosPositionEmbedding(unittest.TestCase):
    """Tests for build_sincos_position_embedding."""

    def test_unsupported_spatial_dims(self):
        """Check that an unsupported spatial_dims raises an error reporting the value."""
        with self.assertRaisesRegex(NotImplementedError, "Spatial Dimension Size 4 Not Implemented"):
            build_sincos_position_embedding(grid_size=[2, 2, 2, 2], embed_dim=8, spatial_dims=4)


if __name__ == "__main__":
    unittest.main()

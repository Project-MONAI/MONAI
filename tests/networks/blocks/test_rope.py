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

import torch
from parameterized import parameterized

from monai.networks.blocks.rope import SpatialRotaryEmbedding, apply_rotary_embedding

TEST_CASES = [[66, (4, 4, 4)], [72, (2, 3, 5)], [64, (4, 8)], [16, (10,)]]


class TestSpatialRotaryEmbedding(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_shape(self, head_dim, feat_shape):
        emb = SpatialRotaryEmbedding(head_dim, feat_shape)()
        num_tokens = int(torch.tensor(feat_shape).prod())
        self.assertEqual(emb.shape, (num_tokens, 2 * head_dim))

    @parameterized.expand(TEST_CASES)
    def test_rotation_preserves_norm(self, head_dim, feat_shape):
        emb = SpatialRotaryEmbedding(head_dim, feat_shape)()
        x = torch.randn(2, 3, emb.shape[0], head_dim)
        out = apply_rotary_embedding(x, emb)
        self.assertEqual(out.shape, x.shape)
        torch.testing.assert_close(out.norm(dim=-1), x.norm(dim=-1))

    def test_relative_position(self):
        # q.k after rotation depends only on the offset between the two positions
        rope = SpatialRotaryEmbedding(8, (16,))
        emb = rope()
        q, k = torch.randn(8), torch.randn(8)

        def score(i, j):
            return (apply_rotary_embedding(q, emb[i]) * apply_rotary_embedding(k, emb[j])).sum()

        torch.testing.assert_close(score(2, 5), score(9, 12))

    def test_origin_is_identity(self):
        emb = SpatialRotaryEmbedding(12, (3, 3, 3))()
        x = torch.randn(12)
        torch.testing.assert_close(apply_rotary_embedding(x, emb[0]), x)

    def test_not_persistent(self):
        self.assertEqual(len(SpatialRotaryEmbedding(12, (3, 3, 3)).state_dict()), 0)

    def test_ill_arg(self):
        with self.assertRaises(ValueError):
            SpatialRotaryEmbedding(64, (4, 4, 4))
        with self.assertRaises(ValueError):
            SpatialRotaryEmbedding(64, ())


if __name__ == "__main__":
    unittest.main()

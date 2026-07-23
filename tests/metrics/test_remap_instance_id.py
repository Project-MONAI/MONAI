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

from monai.metrics.utils import remap_instance_id

_device = "cuda:0" if torch.cuda.is_available() else "cpu"

TEST_CASES = [
    # (description, input, by_size, expected)
    ["non_contiguous_ids", [[0, 2, 2], [5, 5, 0]], False, [[0, 1, 1], [2, 2, 0]]],
    ["already_contiguous", [[0, 1], [2, 2]], False, [[0, 1], [2, 2]]],
    # id 7 covers 3 pixels, id 3 covers 2, id 9 covers 1 -> sizes decide new ids
    ["by_size_largest_first", [[7, 7, 7, 0], [3, 3, 9, 0]], True, [[1, 1, 1, 0], [2, 2, 3, 0]]],
    # equal sizes: ascending original id order wins (stable tie-breaking)
    ["by_size_ties_stable", [[5, 5, 0], [2, 2, 0]], True, [[2, 2, 0], [1, 1, 0]]],
    ["no_background", [[4, 4], [6, 6]], False, [[1, 1], [2, 2]]],
    ["single_instance", [[0, 0], [3, 3]], True, [[0, 0], [1, 1]]],
]


def _reference_remap(pred: torch.Tensor, by_size: bool = False) -> torch.Tensor:
    """The original per-instance-loop implementation, kept as the behavioral reference."""
    pred_id = [i for i in pred.unique() if i != 0]
    if not pred_id:
        return pred
    if by_size:
        instance_size = [(pred == instance_id).sum() for instance_id in pred_id]
        pair_list = sorted(zip(pred_id, instance_size, strict=True), key=lambda x: x[1], reverse=True)
        pred_id = [p[0] for p in pair_list]
    new_pred = torch.zeros_like(pred, dtype=torch.int)
    for idx, instance_id in enumerate(pred_id):
        new_pred[pred == instance_id] = idx + 1
    return new_pred


class TestRemapInstanceId(unittest.TestCase):
    """Tests for `remap_instance_id` covering expected values, pass-through cases, and reference equivalence."""

    @parameterized.expand(TEST_CASES)
    def test_expected_value(self, _, pred, by_size, expected):
        """Remapping produces the hand-computed contiguous ids, including `by_size` ordering and ties."""
        result = remap_instance_id(torch.as_tensor(pred, device=_device), by_size=by_size)
        torch.testing.assert_close(result.cpu(), torch.as_tensor(expected, dtype=torch.int), check_dtype=False)

    @parameterized.expand([["all_background_2d", (4, 4)], ["all_background_3d", (2, 3, 4)], ["empty", (0,)]])
    def test_passthrough(self, _, shape):
        """Inputs without foreground ids are returned unchanged, keeping their original dtype."""
        pred = torch.zeros(shape, dtype=torch.int64, device=_device)
        result = remap_instance_id(pred, by_size=True)
        self.assertEqual(result.dtype, pred.dtype)
        torch.testing.assert_close(result, pred)

    def test_output_dtype(self):
        """Remapped outputs use the `torch.int` dtype regardless of input dtype."""
        pred = torch.as_tensor([[0, 9]], dtype=torch.int64, device=_device)
        self.assertEqual(remap_instance_id(pred).dtype, torch.int)

    @parameterized.expand(
        [
            ["2d", (64, 64), 20, False],
            ["2d_by_size", (64, 64), 20, True],
            ["3d_by_size", (16, 16, 16), 12, True],
            ["sparse_ids_by_size", (48, 48), 7, True],
        ]
    )
    def test_matches_reference(self, name, shape, n_inst, by_size):
        """Randomized inputs produce output identical to the previous per-instance-loop implementation."""
        generator = torch.Generator().manual_seed(0)
        pred = torch.randint(0, n_inst + 1, shape, generator=generator).to(_device)
        if name.startswith("sparse"):
            pred = pred * 1000 + 17  # large, non-contiguous, no-background ids
        result = remap_instance_id(pred, by_size=by_size)
        expected = _reference_remap(pred, by_size=by_size)
        torch.testing.assert_close(result, expected)


if __name__ == "__main__":
    unittest.main()

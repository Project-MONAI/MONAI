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
from parameterized.parameterized import parameterized

from monai.data import MetaTensor, set_track_meta
from monai.transforms.croppad.array import Pad

# "symmetric" routes through the NumPy backend, which pads the channel dim.
CHANNEL_PAD_CASES = [
    [[(2, 0), (0, 0), (0, 0)]],
    [[(0, 2), (0, 0), (0, 0)]],
    [[(2, 2), (0, 0), (0, 0)]],
    [[(1, 3), (1, 1), (0, 2)]],
]


class TestPadInverseChannel(unittest.TestCase):
    @parameterized.expand(CHANNEL_PAD_CASES)
    def test_inverse_roundtrip_channel_pad(self, to_pad):
        set_track_meta(True)
        img = MetaTensor(torch.arange(3 * 4 * 4).reshape(3, 4, 4).float())
        pad = Pad(to_pad=to_pad, mode="symmetric")
        padded = pad(img.clone())
        self.assertEqual(padded.shape[0], img.shape[0] + to_pad[0][0] + to_pad[0][1])
        inv = pad.inverse(padded)
        self.assertEqual(inv.shape, img.shape)
        self.assertTrue(torch.equal(inv.as_tensor(), img.as_tensor()))


if __name__ == "__main__":
    unittest.main()

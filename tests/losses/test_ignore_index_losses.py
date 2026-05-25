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

from monai.losses import AsymmetricUnifiedFocalLoss, DiceLoss, FocalLoss, TverskyLoss

# Defining test cases: (LossClass, args)
TEST_CASES = [
    (DiceLoss, {"sigmoid": True}),
    (FocalLoss, {"use_softmax": False}),
    (TverskyLoss, {"sigmoid": True}),
    (AsymmetricUnifiedFocalLoss, {}),
]

CLASS_INDEX_TEST_CASES = [
    (DiceLoss, {"softmax": True, "to_onehot_y": True}),
    (FocalLoss, {"use_softmax": True, "to_onehot_y": True}),
    (TverskyLoss, {"softmax": True, "to_onehot_y": True}),
    (AsymmetricUnifiedFocalLoss, {"to_onehot_y": True}),
]

SENTINEL_ONEHOT_TEST_CASES = [
    (DiceLoss, {"softmax": True}),
    (FocalLoss, {"use_softmax": True}),
    (TverskyLoss, {"softmax": True}),
]


class TestIgnoreIndexLosses(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_loss_ignore_consistency(self, loss_class, kwargs):
        ignore_index = 255
        loss_func = loss_class(ignore_index=ignore_index, **kwargs)

        # Create two inputs that are identical EXCEPT in the area designated as 'ignored'
        # Input shape: [Batch, Channel, H, W]
        input_base = torch.randn(1, 1, 4, 4)
        input_alt = input_base.clone()
        input_alt[0, 0, 2:, :] += 5.0  # Significant difference in the bottom half

        # Target: Top half is valid (0,1), Bottom half is ignored (255)
        target = torch.tensor(
            [[[[1, 0, 1, 0], [0, 1, 0, 1], [255, 255, 255, 255], [255, 255, 255, 255]]]], dtype=torch.float
        )

        # Execute
        loss_base = loss_func(input_base, target)
        loss_alt = loss_func(input_alt, target)

        # ASSERTION: The losses must be identical because the difference
        # occurred only in the ignored region.
        torch.testing.assert_close(loss_base, loss_alt, atol=1e-5, rtol=1e-5)

    @parameterized.expand(TEST_CASES)
    def test_no_ignore_behavior(self, loss_class, kwargs):
        # Ensure that when ignore_index is None, the loss functions normally
        loss_func = loss_class(ignore_index=None, **kwargs)
        input_data = torch.randn(1, 1, 4, 4)
        target = torch.randint(0, 2, (1, 1, 4, 4)).float()

        output = loss_func(input_data, target)
        self.assertFalse(torch.isnan(output))

    @parameterized.expand([(loss_class, kwargs, ignore_index) for loss_class, kwargs in CLASS_INDEX_TEST_CASES for ignore_index in (0, 1)])
    def test_loss_ignore_class_index(self, loss_class, kwargs, ignore_index):
        loss_func = loss_class(ignore_index=ignore_index, **kwargs)

        ignored_rows = slice(0, 2) if ignore_index == 0 else slice(2, 4)
        ignored_channel = ignore_index

        input_base = torch.randn(1, 2, 4, 4)
        input_alt = input_base.clone()
        input_alt[:, ignored_channel, ignored_rows, :] += 5.0

        target = torch.zeros((1, 1, 4, 4), dtype=torch.long)
        target[:, 0, 0:2, :] = 0
        target[:, 0, 2:, :] = 1

        loss_base = loss_func(input_base, target)
        loss_alt = loss_func(input_alt, target)

        torch.testing.assert_close(loss_base, loss_alt, atol=1e-5, rtol=1e-5)

    @parameterized.expand(SENTINEL_ONEHOT_TEST_CASES)
    def test_loss_ignore_sentinel_onehot(self, loss_class, kwargs):
        ignore_index = 255
        loss_func = loss_class(ignore_index=ignore_index, to_onehot_y=True, **kwargs)

        input_base = torch.randn(1, 3, 4, 4)
        input_alt = input_base.clone()
        input_alt[:, 1, 2:, :] += 5.0

        target = torch.tensor(
            [[[[1, 0, 1, 0], [0, 1, 0, 1], [255, 255, 255, 255], [255, 255, 255, 255]]]], dtype=torch.long
        )

        loss_base = loss_func(input_base, target)
        loss_alt = loss_func(input_alt, target)

        torch.testing.assert_close(loss_base, loss_alt, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()

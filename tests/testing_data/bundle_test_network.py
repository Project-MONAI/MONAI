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

import torch

from monai.networks.nets import UNet


class TestMultiInputUNet(UNet):
    """
    This class is used for "tests/test_bundle_verify_net.py" to show that the monai.bundle.verify_net_in_out
    function supports to verify networks that have multiple args as the input in the forward function.
    """

    def forward(self, x: torch.Tensor, extra_arg1: int, extra_arg2: int) -> torch.Tensor:  # type: ignore
        x = self.model(x)
        x += extra_arg1
        x += extra_arg2
        return x

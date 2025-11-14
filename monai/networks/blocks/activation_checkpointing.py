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

from typing import cast

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class ActivationCheckpointWrapper(nn.Module):
    """Wrapper applying activation checkpointing to a module during training.

    Args:
        module: The module to wrap with activation checkpointing.
    """

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional activation checkpointing.

        Args:
            x: Input tensor.

        Returns:
            Output tensor from the wrapped module.
        """
        return cast(torch.Tensor, checkpoint(self.module, x, use_reentrant=False))

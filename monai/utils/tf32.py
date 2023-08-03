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

import os
import warnings

import torch

from monai.utils.module import pytorch_after, version_geq

__all__ = ["has_ampere_or_later", "detect_default_tf32"]


def has_ampere_or_later() -> bool:
    """
    Check if there is any Ampere and later GPU.
    """
    if not torch.cuda.is_available():
        return False
    if not version_geq(f"{torch.version.cuda}", "11.0"):
        return False
    for i in range(torch.cuda.device_count()):
        major, _ = torch.cuda.get_device_capability(i)
        if major >= 8:  # Ampere and later
            return True
    return False


def detect_default_tf32() -> bool:
    """
    Dectect if there is anything that may enable TF32 mode by default.
    If any, show a warning message.
    """
    may_enable_tf32 = False
    try:
        if not has_ampere_or_later():
            return False

        if pytorch_after(1, 7, 0) and not pytorch_after(1, 12, 0):
            warnings.warn(
                "torch.backends.cuda.matmul.allow_tf32 = True by default.\n"
                "  This value defaults to True when PyTorch version in [1.7, 1.11] and may affect precision\n"
                "  See https://docs.monai.io/en/latest/precision_accelerating.html#precision-and-accelerating"
            )
            may_enable_tf32 = True

        override_tf32_env_vars = {"NVIDIA_TF32_OVERRIDE": "1", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": "1"}
        for name, override_val in override_tf32_env_vars.items():
            if os.environ.get(name) == override_val:
                warnings.warn(
                    f"Environment variable `{name} = {override_val}` is set.\n"
                    f"  This environment variable may enable TF32 mode accidentally and affect precision.\n"
                    f"  See https://docs.monai.io/en/latest/precision_accelerating.html#precision-and-accelerating"
                )
                may_enable_tf32 = True

        return may_enable_tf32
    except:
        return False

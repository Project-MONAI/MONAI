# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable, Dict, Optional, Type

import torch.nn as nn

from .cast_utils import CastToFloat

def simple_replace(
    BaseT: Type[nn.Module], DestT: Type[nn.Module]
) -> Callable[[nn.Module], Optional[nn.Module]]:
    """
    Generic function generator to replace BaseT module with DestT. BaseT and DestT should have same atrributes. No weights are copied.
    Args:
        BaseT : module type to replace
        DestT : destination module type
    Returns:
        swap function to replace BaseT module with DestT
    """

    def expansion_fn(mod: nn.Module) -> Optional[nn.Module]:
        if not isinstance(mod, BaseT):
            return None
        args = [getattr(mod, name, None) for name in mod.__constants__]
        out = DestT(*args)
        return out

    return expansion_fn


def wrap_module(
    BaseT: Type[nn.Module], DestT: Type[nn.Module]
) -> Callable[[nn.Module], Optional[nn.Module]]:
    """
    Generic function generator to replace BaseT module with DestT wrapper.
    Args:
        BaseT : module type to replace
        DestT : destination module type
    Returns:
        swap function to replace BaseT module with DestT
    """

    def expansion_fn(mod: nn.Module) -> Optional[nn.Module]:
        out = DestT(mod)
        return out

    return expansion_fn


def swap_modules(model: nn.Module, mapping: Dict[str, nn.Module]) -> nn.Module:
    """
    This function swaps nested modules as specified by "dot paths" in mod with a desired replacement. This allows
    for swapping nested modules through arbitrary levels if children

    NOTE: This occurs in place, if you want to preserve model then make sure to copy it first.

    """
    for path, new_mod in mapping.items():
        expanded_path = path.split(".")
        parent_mod = model
        for sub_path in expanded_path[:-1]:
            parent_mod = parent_mod._modules[sub_path]
        parent_mod._modules[expanded_path[-1]] = new_mod

    return model


def replace_modules(
    model: nn.Module,
    expansions: Dict[str, Callable[[nn.Module], Optional[nn.Module]]],
) -> nn.Module:
    """
    Top-level function to replace modules in model, specified by class name with a desired replacement.
    NOTE: This occurs in place, if you want to preserve model then make sure to copy it first.
    Args:
        model : top level module
        expansions : replacement dictionary: module class name -> replacement function generator
    Returns:
        model, possibly modified in-place
    """
    mapping: Dict[str, nn.Module] = {}
    for name, m in model.named_modules():
        m_type = type(m).__name__
        if m_type in expansions:
            # print (f"Found {m_type} in expansions ...")
            swapped = expansions[m_type](m)
            if swapped:
                mapping[name] = swapped

    print(f"Swapped {len(mapping)} modules")
    swap_modules(model, mapping)
    return model

def replace_for_export(model: nn.Module, do_cast: bool = True) -> nn.Module:
    """
    Top-level function to replace default set of modules in model
    NOTE: This occurs in place, if you want to preserve model then make sure to copy it first.
    Args:
        model : top level module
    Returns:
        model, possibly modified in-place
    """
    if do_cast:
        print("Adding casts around norms...")
        cast_replacements = {
            "BatchNorm1d": wrap_module(nn.BatchNorm1d, CastToFloat),
            "BatchNorm2d": wrap_module(nn.BatchNorm2d, CastToFloat),
            "BatchNorm3d": wrap_module(nn.BatchNorm2d, CastToFloat),
            "LayerNorm": wrap_module(nn.LayerNorm, CastToFloat),
            "InstanceNorm1d": wrap_module(nn.InstanceNorm1d, CastToFloat),
            "InstanceNorm3d": wrap_module(nn.InstanceNorm3d, CastToFloat),
        }
        replace_modules(model, cast_replacements)
    return model

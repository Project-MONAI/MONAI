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

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cast_utils import CastToFloat


class LinearWithBiasSkip(nn.Module):
    def __init__(self, weight, bias, skip_bias_add):
        super(LinearWithBiasSkip, self).__init__()
        self.bias = bias
        self.weight = weight
        self.skip_bias_add = skip_bias_add

    def forward(self, x):
        if self.skip_bias_add:
            return F.linear(x, self.weight), self.bias
        return F.linear(x, self.weight, self.bias), None

apex_available = True

try:
    from apex.contrib.layer_norm.layer_norm import FastLayerNorm
    from apex.normalization.fused_layer_norm import FusedLayerNorm, MixedFusedLayerNorm
    from apex.transformer.functional.fused_softmax import FusedScaleMaskSoftmax
    from apex.transformer.tensor_parallel.layers import (
        ColumnParallelLinear,
        RowParallelLinear,
    )

    def replace_FusedLayerNorm(n: nn.Module) -> Optional[nn.LayerNorm]:
        """
        Replaces Apex's FusedLayerNorm with nn.LayerNorm. This is required for ONNX export.
        Args:
           n: the FusedLayerNorm pytorch module to replace
        Returns:
           Equivalent LayerNorm module
        """

        p = next(n.parameters())
        if isinstance(n, FusedLayerNorm) or isinstance(n, MixedFusedLayerNorm):
            shape, eps, affine = n.normalized_shape, n.eps, n.elementwise_affine
        elif isinstance(n, FastLayerNorm):
            shape, eps, affine = n.weight.shape, n.epsilon, True
        else:
            return None

        mod = nn.LayerNorm(
            shape, eps=eps, elementwise_affine=affine, device=p.device, dtype=p.dtype
        )
        n_state = n.state_dict()
        mod.load_state_dict(n_state)
        return mod

    def replace_RowParallelLinear(n: nn.Module) -> Optional[nn.Linear]:
        """
        Replaces Apex's FusedLayerNorm with nn.LayerNorm. This is required for ONNX export.
        Args:
           n: the FusedLayerNorm pytorch module to replace
        Returns:
           Equivalent LayerNorm module
        """
        if not isinstance(n, RowParallelLinear):
            raise ValueError(
                "This function can only change the RowParallelLinear module."
            )

        dev = next(n.parameters()).device
        mod = LinearWithBiasSkip(n.weight, n.bias, n.skip_bias_add).to(device=dev)

        n_state = n.state_dict()
        mod.load_state_dict(n_state)
        return mod

    def replace_ParallelLinear(n: nn.Module) -> Optional[nn.Linear]:
        """
        Replaces Apex's ColumnParallelLinear or RowParallelLinear with nn.Linear
        Args:
           n: the nn.Module pytorch module to replace
        Returns:
           Equivalent Linear module
        """
        if not (
            isinstance(n, ColumnParallelLinear) or isinstance(n, RowParallelLinear)
        ):
            raise ValueError(
                "This function can only change the ColumnParallelLinear or RowParallelLinear module."
            )

        dev = next(n.parameters()).device
        mod = LinearWithBiasSkip(n.weight, n.bias, n.skip_bias_add).to(dev)

        n_state = n.state_dict()
        mod.load_state_dict(n_state)
        return mod

    def replace_FusedScaleMaskSoftmax(n: nn.Module) -> Optional[nn.Linear]:
        """
        Replaces Apex's FusedScaleMaskSoftmax with nn.LayerNorm. This is required for ONNX export.
        Args:
           n: the FusedScaleMaskSoftmax module to replace
        Returns:
           Equivalent LayerNorm module
        """
        if not isinstance(n, FusedScaleMaskSoftmax):
            raise ValueError(
                "This function can only change the FusedScaleMaskSoftmax module."
            )

        # disable the fusion only
        mod = FusedScaleMaskSoftmax(
            n.input_in_fp16,
            n.input_in_bf16,
            n.attn_mask_type,
            False,
            n.mask_func,
            n.softmax_in_fp32,
            n.scale,
        )

        return mod

    default_Apex_replacements = {
        "FusedLayerNorm": replace_FusedLayerNorm,
        "MixedFusedLayerNorm": replace_FusedLayerNorm,
        "FastLayerNorm": replace_FusedLayerNorm,
        "ESM1bLayerNorm": replace_FusedLayerNorm,
        "RowParallelLinear": replace_ParallelLinear,
        "ColumnParallelLinear": replace_ParallelLinear,
        "FusedScaleMaskSoftmax": replace_FusedScaleMaskSoftmax,
    }

except Exception:
    default_Apex_replacements = {}
    apex_available = False


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


def replace_MatchedScaleMaskSoftmax(n: nn.Module) -> Optional[nn.Linear]:
    """
    Replaces MatchedScaleMaskSoftmax with exportable softmax layer
    Args:
        n: module to replace
    Returns:
        exportable module
    """
    # including the import here to avoid circular imports
    from nemo.collections.nlp.modules.common.megatron.fused_softmax import (
        MatchedScaleMaskSoftmax,
    )

    # disabling fusion for the MatchedScaleMaskSoftmax
    mod = MatchedScaleMaskSoftmax(
        n.input_in_fp16,
        n.input_in_bf16,
        n.attn_mask_type,
        False,
        n.mask_func,
        n.softmax_in_fp32,
        n.scale,
    )
    return mod


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


def swap_modules(model: nn.Module, mapping: Dict[str, nn.Module]):
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
    expansions: Dict[str, Callable[[nn.Module], Optional[nn.Module]]] = None,
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

def replace_for_export(model: nn.Module, do_cast: bool = False) -> nn.Module:
    """
    Top-level function to replace default set of modules in model
    NOTE: This occurs in place, if you want to preserve model then make sure to copy it first.
    Args:
        model : top level module
        replace_1D_2D : include 1D -> 2D replacements
    Returns:
        model, possibly modified in-place
    """
    if apex_available:
        print("Replacing Apex layers ...")
        replace_modules(model, default_Apex_replacements)

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

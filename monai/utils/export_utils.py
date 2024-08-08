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

from typing import Callable

import torch
import torch.nn as nn


def cast_tensor(x, from_dtype=torch.float16, to_dtype=torch.float32):
    """
    Utility function to cast a single tensor from from_dtype to to_dtype
    """
    return x.to(dtype=to_dtype) if x.dtype == from_dtype else x


def cast_all(x, from_dtype=torch.float16, to_dtype=torch.float32):
    """
    Utility function to cast all tensors in a tuple from from_dtype to to_dtype
    """
    if isinstance(x, torch.Tensor):
        return cast_tensor(x, from_dtype=from_dtype, to_dtype=to_dtype)
    else:
        if isinstance(x, dict):
            new_dict = {}
            for k in x.keys():
                new_dict[k] = cast_all(x[k], from_dtype=from_dtype, to_dtype=to_dtype)
            return new_dict
        elif isinstance(x, tuple):
            return tuple(cast_all(y, from_dtype=from_dtype, to_dtype=to_dtype) for y in x)


class CastToFloat(torch.nn.Module):
    """
    Class used to add autocast protection for ONNX export
    for forward methods with single return vaue
    """

    def __init__(self, mod):
        super().__init__()
        self.mod = mod

    def forward(self, x):
        dtype = x.dtype
        with torch.cuda.amp.autocast(enabled=False):
            ret = self.mod.forward(x.to(torch.float32)).to(dtype)
        return ret


class CastToFloatAll(torch.nn.Module):
    """
    Class used to add autocast protection for ONNX export
    for forward methods with multiple return values
    """

    def __init__(self, mod):
        super().__init__()
        self.mod = mod

    def forward(self, *args):
        from_dtype = args[0].dtype
        with torch.cuda.amp.autocast(enabled=False):
            ret = self.mod.forward(*cast_all(args, from_dtype=from_dtype, to_dtype=torch.float32))
        return cast_all(ret, from_dtype=torch.float32, to_dtype=from_dtype)


def simple_replace(base_t: type[nn.Module], dest_t: type[nn.Module]) -> Callable[[nn.Module], nn.Module | None]:
    """
    Generic function generator to replace base_t module with dest_t.
    base_t and dest_t should have same atrributes. No weights are copied.
    Args:
        base_t : module type to replace
        dest_t : destination module type
    Returns:
        swap function to replace base_t module with dest_t
    """

    def expansion_fn(mod: nn.Module) -> nn.Module | None:
        if not isinstance(mod, base_t):
            return None
        args = [getattr(mod, name, None) for name in mod.__constants__]
        out = dest_t(*args)
        return out

    return expansion_fn


def wrap_module(base_t: type[nn.Module], dest_t: type[nn.Module]) -> Callable[[nn.Module], nn.Module | None]:
    """
    Generic function generator to replace base_t module with dest_t wrapper.
    Args:
        base_t : module type to replace
        dest_t : destination module type
    Returns:
        swap function to replace base_t module with dest_t
    """

    def expansion_fn(mod: nn.Module) -> nn.Module | None:
        out = dest_t(mod)
        return out

    return expansion_fn


def swap_modules(model: nn.Module, mapping: dict[str, nn.Module]) -> nn.Module:
    """
    This function swaps nested modules as specified by "dot paths" in mod with a desired replacement. This allows
    for swapping nested modules through arbitrary levels if children

    NOTE: This occurs in place, if you want to preserve model then make sure to copy it first.

    """
    for path, new_mod in mapping.items():
        expanded_path = path.split(".")
        parent_mod = model
        for sub_path in expanded_path[:-1]:
            submod = parent_mod._modules[sub_path]
            if submod is None:
                break
            else:
                parent_mod = submod
        parent_mod._modules[expanded_path[-1]] = new_mod

    return model


def replace_modules(model: nn.Module, expansions: dict[str, Callable[[nn.Module], nn.Module | None]]) -> nn.Module:
    """
    Top-level function to replace modules in model, specified by class name with a desired replacement.
    NOTE: This occurs in place, if you want to preserve model then make sure to copy it first.
    Args:
        model : top level module
        expansions : replacement dictionary: module class name -> replacement function generator
    Returns:
        model, possibly modified in-place
    """
    mapping: dict[str, nn.Module] = {}
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

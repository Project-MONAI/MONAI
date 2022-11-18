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
"""
Decorators and context managers for NVIDIA Tools Extension to profile MONAI components
"""

from collections import defaultdict
from functools import wraps
from typing import Any, Optional, Tuple, Union

from torch.autograd import Function
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import Dataset

from monai.utils import ensure_tuple, optional_import

_nvtx, _ = optional_import("torch._C._nvtx", descriptor="NVTX is not installed. Are you sure you have a CUDA build?")

__all__ = ["Range"]


class Range:
    """
    A decorator and context manager for NVIDIA Tools Extension (NVTX) Range for profiling.
    When used as a decorator it encloses a specific method of the object with an NVTX Range.
    When used as a context manager, it encloses the runtime context (created by with statement) with an NVTX Range.

    Args:
        name: the name to be associated to the range
        methods: (only when used as decorator) the name of a method (or a list of the name of the methods)
            to be wrapped by NVTX range.
            If None (default), the method(s) will be inferred based on the object's type for various MONAI components,
            such as Networks, Losses, Functions, Transforms, and Datasets.
            Otherwise, it look up predefined methods: "forward", "__call__", "__next__", "__getitem__"
        append_method_name: if append the name of the methods to be decorated to the range's name
            If None (default), it appends the method's name only if we are annotating more than one method.
        recursive: if set to True, it will recursively annotate every individual module in a list
            or in a chain of modules (chained using Compose). Default to False.

    """

    name_counter: dict = defaultdict(int)

    def __init__(
        self,
        name: Optional[str] = None,
        methods: Optional[Union[str, Tuple[str, ...]]] = None,
        append_method_name: Optional[bool] = None,
        recursive: bool = False,
    ) -> None:
        self.name = name
        self.methods = methods
        self.append_method_name = append_method_name
        self.recursive = recursive

    def __call__(self, obj: Any):
        if self.recursive is True:
            if isinstance(obj, (list, tuple)):
                return type(obj)(Range(recursive=True)(t) for t in obj)

            from monai.transforms.compose import Compose

            if isinstance(obj, Compose):
                obj.transforms = Range(recursive=True)(obj.transforms)

            self.recursive = False

        # Define the name to be associated to the range if not provided
        if self.name is None:
            name = type(obj).__name__
            # If CuCIM or TorchVision transform wrappers are being used,
            # append the underlying transform to the name for more clarity
            if "CuCIM" in name or "TorchVision" in name:
                name = f"{name}_{obj.name}"
            self.name_counter[name] += 1
            if self.name_counter[name] > 1:
                self.name = f"{name}_{self.name_counter[name]}"
            else:
                self.name = name

        # Define the methods to be wrapped if not provided
        if self.methods is None:
            self.methods = self._get_method(obj)
        else:
            self.methods = ensure_tuple(self.methods)

        # Check if to append method's name to the range's name
        if self.append_method_name is None:
            if len(self.methods) > 1:
                self.append_method_name = True
            else:
                self.append_method_name = False

        # Decorate the methods
        for method in self.methods:
            self._decorate_method(obj, method, self.append_method_name)

        return obj

    def _decorate_method(self, obj, method, append_method_name):
        # Append the method's name to the range's name
        if append_method_name:
            name = f"{self.name}.{method}"
        else:
            name = self.name

        # Get the class for special functions
        if method.startswith("__"):
            owner = type(obj)
        else:
            owner = obj

        # Get the method to be wrapped
        _temp_func = getattr(owner, method)

        # Wrap the method with NVTX range (range push/pop)
        @wraps(_temp_func)
        def range_wrapper(*args, **kwargs):
            _nvtx.rangePushA(name)
            output = _temp_func(*args, **kwargs)
            _nvtx.rangePop()
            return output

        # Replace the method with the wrapped version
        if method.startswith("__"):
            # If it is a special method, it requires special attention
            class NVTXRangeDecoratedClass(owner):
                ...

            setattr(NVTXRangeDecoratedClass, method, range_wrapper)
            obj.__class__ = NVTXRangeDecoratedClass

        else:
            setattr(owner, method, range_wrapper)

    def _get_method(self, obj: Any) -> tuple:
        if isinstance(obj, Module):
            method_list = ["forward"]
        elif isinstance(obj, Optimizer):
            method_list = ["step"]
        elif isinstance(obj, Function):
            method_list = ["forward", "backward"]
        elif isinstance(obj, Dataset):
            method_list = ["__getitem__"]
        else:
            default_methods = ["forward", "__call__", "__next__", "__getitem__"]
            method_list = []
            for method in default_methods:
                if hasattr(obj, method):
                    method_list.append(method)
            if len(method_list) < 1:
                raise ValueError(
                    f"The method to be wrapped for this object [{type(obj)}] is not recognized."
                    "The name of the method should be provided or the object should have one of these methods:"
                    f"{default_methods}"
                )
        return ensure_tuple(method_list)

    def __enter__(self):
        if self.name is None:
            # Number the range with class variable counter to avoid duplicate names.
            self.name_counter["context"] += 1
            self.name = f"context_{self.name_counter['context']}"

        _nvtx.rangePushA(self.name)

    def __exit__(self, type, value, traceback):
        _nvtx.rangePop()

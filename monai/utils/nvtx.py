# Copyright 2020 - 2021 MONAI Consortium
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
from typing import Any, Optional

import torch.nn as nn

from monai.utils import optional_import

_nvtx, _ = optional_import("torch._C._nvtx", descriptor="NVTX is not installed. Are you sure you have a CUDA build?")

__all__ = ["Range"]


class Range:
    """
    A decorator and context manager for NVIDIA Tools Extension (NVTX) Range for profiling.
    When used as a decorator it encloses a specific method of the object with an NVTX Range.
    When used as a context manager, it encloses the runtime context (created by with statement) with an NVTX Range.

    Args:
        name: the name to be associated to the range
        method: (only when used as decorator) the method to be wrapped by NVTX range. If not provided (None),
            the method will be inferred based on the object's class for Callable objects (like Transforms),
            nn.Module objects (like Networks, Losses, etc.), and Dataloaders. Method resolve order is:
            - forward()
            - __call__()
            - __next__()

    """

    name_counter: dict = defaultdict(int)

    def __init__(self, name: Optional[str] = None, method: Optional[str] = None) -> None:
        self.name = name
        self.method = method

    def __call__(self, obj: Any):
        # Define the name to be associated to the range if not provided
        if self.name is None:
            name = type(obj).__name__
            self.name_counter[name] += 1
            self.name = f"{name}_{self.name_counter[name]}"

        # Define the method to be wrapped if not provided
        if self.method is None:
            method_list = [
                "forward",  # nn.Module
                "__call__",  # Callable
                "__next__",  # Dataloader
            ]
            for method in method_list:
                if hasattr(obj, method):
                    self.method = method
                    break
            if self.method is None:
                raise ValueError(
                    f"The method to be wrapped for this object [{type(obj)}] is not recognized."
                    "The name of the method should be provied or the object should have one of these methods:"
                    f"{method_list}"
                )

        # Get the class for special functions
        if self.method.startswith("__"):
            owner = type(obj)
        else:
            owner = obj

        # Get the method to be wrapped
        _temp_func = getattr(owner, self.method)

        # Wrap the method with NVTX range (range push/pop)
        @wraps(_temp_func)
        def range_wrapper(*args, **kwargs):
            _nvtx.rangePushA(self.name)
            output = _temp_func(*args, **kwargs)
            _nvtx.rangePop()
            return output

        # Replace the method with the wrapped version
        setattr(owner, self.method, range_wrapper)

        return obj

    def __enter__(self):
        if self.name is None:
            # Number the range with class variable counter to avoid duplicate names.
            self.name_counter["context"] += 1
            self.name = f"context_{self.name_counter['context']}"

        _nvtx.rangePushA(self.name)

    def __exit__(self, type, value, traceback):
        _nvtx.rangePop()

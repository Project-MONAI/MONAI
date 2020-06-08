# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import itertools
from collections.abc import Iterable
from collections import OrderedDict
import copy
from typing import Optional, Dict, Any, List

import numpy as np
import torch
import random

_seed = None


def zip_with(op, *vals, mapfunc=map):
    """
    Map `op`, using `mapfunc`, to each tuple derived from zipping the iterables in `vals`.
    """
    return mapfunc(op, zip(*vals))


def star_zip_with(op, *vals):
    """
    Use starmap as the mapping function in zipWith.
    """
    return zip_with(op, *vals, mapfunc=itertools.starmap)


def first(iterable, default=None):
    """
    Returns the first item in the given iterable or `default` if empty, meaningful mostly with 'for' expressions.
    """
    for i in iterable:
        return i
    return default


def issequenceiterable(obj):
    """
    Determine if the object is an iterable sequence and is not a string
    """
    return isinstance(obj, Iterable) and not isinstance(obj, str)


def ensure_tuple(vals):
    """Returns a tuple of `vals`"""
    if not issequenceiterable(vals):
        vals = (vals,)

    return tuple(vals)


def ensure_tuple_size(tup, dim):
    """Returns a copy of `tup` with `dim` values by either shortened or padded with zeros as necessary."""
    tup = tuple(tup) + (0,) * dim
    return tup[:dim]


def ensure_tuple_rep(tup, dim):
    """
    Returns a copy of `tup` with `dim` values by either shortened or duplicated input.
    """
    if not issequenceiterable(tup):
        return (tup,) * dim
    elif len(tup) == dim:
        return tuple(tup)

    raise ValueError(f"sequence must have length {dim}, got length {len(tup)}.")


def is_scalar_tensor(val):
    if torch.is_tensor(val) and val.ndim == 0:
        return True
    return False


def is_scalar(val):
    if torch.is_tensor(val) and val.ndim == 0:
        return True
    return np.isscalar(val)


def process_bar(index: int, count: int, bar_len: int = 30, newline: bool = False):
    """print a process bar to track some time consuming task.

    Args:
        index (int): current satus in process.
        count (int): total steps of the process.
        bar_len(int): the total length of the bar on screen, default is 30 char.
        newline (bool): whether to print in a new line for every index.
    """
    end = "\r" if newline is False else "\r\n"
    filled_len = int(bar_len * index // count)
    bar = "[" + "=" * filled_len + " " * (bar_len - filled_len) + "]"
    print(f"{index}/{count} {bar:s}  ", end=end)
    if index == count:
        print("")


def get_seed():
    return _seed


def set_determinism(seed=np.iinfo(np.int32).max, additional_settings=None):
    """
    Set random seed for modules to enable or disable deterministic training.

    Args:
        seed (None, int): the random seed to use, default is np.iinfo(np.int32).max.
            It is recommended to set a large seed, i.e. a number that has a good balance
            of 0 and 1 bits. Avoid having many 0 bits in the seed.
            if set to None, will disable deterministic training.
        additional_settings (Callable, list or tuple of Callables): additional settings
            that need to set random seed.

    """
    if seed is None:
        # cast to 32 bit seed for CUDA
        seed_ = torch.default_generator.seed() % (np.iinfo(np.int32).max + 1)
        if not torch.cuda._is_in_bad_fork():
            torch.cuda.manual_seed_all(seed_)
    else:
        torch.manual_seed(seed)

    global _seed
    _seed = seed
    random.seed(seed)
    np.random.seed(seed)

    if additional_settings is not None:
        additional_settings = ensure_tuple(additional_settings)
        for func in additional_settings:
            func(seed)

    if seed is not None:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False


def validate_kwargs(args: Iterable, input_kwargs: Dict[str, Any], reference_args: OrderedDict) -> Dict[str, Any]:
    """
    A function to validate dictionary arguments for a function.  Used to provide a common
    base class with subclasses accepting different arguments.
    Args:
        input_kwargs:  **kwargs like dictionary with string based keys
        reference_args: Ordered dictionary of valid arguments and default values to be used,
                        the string value of "__from_input__" is used to indicate that the
                        value must be provided from the input.
    Returns:
        Validated and default value supplied dictionary.
    """
    assert isinstance(reference_args, OrderedDict), f"reference_args must be an OrderedDict, is {type(reference_args)}"
    output_args = copy.deepcopy(reference_args)
    unnamed_args_dictionary: OrderedDict = OrderedDict(zip(reference_args.keys(), args))
    input_kwargs.update(unnamed_args_dictionary)
    input_dictionary_keys: set = set(input_kwargs.keys())

    set_diff_valid_keys = input_dictionary_keys - set(output_args.keys())
    if set_diff_valid_keys:
        raise KeyError(f"Invalid arguments provided: {set_diff_valid_keys}")
    required_keys: List[str] = [k for (k, v) in output_args.items() if v == "__from_input__"]
    set_diff_required_keys = set(required_keys) - input_dictionary_keys
    if set_diff_required_keys:
        raise KeyError(f"Missing mandatory keys: {set_diff_required_keys}")
    output_args.update(input_kwargs)
    return output_args

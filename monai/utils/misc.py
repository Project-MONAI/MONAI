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
from collections.abc import Iterable, Sequence
from typing import Any, Tuple, Callable

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


def issequenceiterable(obj) -> bool:
    """
    Determine if the object is an iterable sequence and is not a string.
    """
    return isinstance(obj, Iterable) and not isinstance(obj, str)


def ensure_tuple(vals: Any) -> Tuple:
    """
    Returns a tuple of `vals`.
    """
    if not issequenceiterable(vals):
        vals = (vals,)

    return tuple(vals)


def ensure_tuple_size(tup, dim: int, pad_val=0) -> Tuple:
    """
    Returns a copy of `tup` with `dim` values by either shortened or padded with `pad_val` as necessary.
    """
    tup = ensure_tuple(tup) + (pad_val,) * dim
    return tuple(tup[:dim])


def ensure_tuple_rep(tup: Any, dim: int):
    """
    Returns a copy of `tup` with `dim` values by either shortened or duplicated input.

    Examples::

        >>> ensure_tuple_rep(1, 3)
        (1, 1, 1)
        >>> ensure_tuple_rep(None, 3)
        (None, None, None)
        >>> ensure_tuple_rep('test', 3)
        ('test', 'test', 'test')
        >>> ensure_tuple_rep([1, 2, 3], 3)
        (1, 2, 3)
        >>> ensure_tuple_rep(range(3), 3)
        (0, 1, 2)
        >>> ensure_tuple_rep([1, 2], 3)
        ValueError: sequence must have length 3, got length 2.

    Raises:
        ValueError: sequence must have length {dim}, got length {len(tup)}.

    """
    if not issequenceiterable(tup):
        return (tup,) * dim
    elif len(tup) == dim:
        return tuple(tup)

    raise ValueError(f"sequence must have length {dim}, got length {len(tup)}.")


def fall_back_tuple(user_provided: Any, default: Sequence, func: Callable = lambda x: x and x > 0) -> Tuple:
    """
    Refine `user_provided` according to the `default`, and returns as a validated tuple.

    The validation is done for each element in `user_provided` using `func`.
    If `func(user_provided[idx])` returns False, the corresponding `default[idx]` will be used
    as the fallback.

    Typically used when `user_provided` is a tuple of window size provided by the user,
    `default` is defined by data, this function returns an updated `user_provided` with its non-positive
    components replaced by the corresponding components from `default`.

    Args:
        user_provided: item to be validated.
        default: a sequence used to provided the fallbacks.
        func: a Callable to validate every components of `user_provided`.

    Examples::

        >>> fall_back_tuple((1, 2), (32, 32))
        (1, 2)
        >>> fall_back_tuple(None, (32, 32))
        (32, 32)
        >>> fall_back_tuple((-1, 10), (32, 32))
        (32, 10)
        >>> fall_back_tuple((-1, None), (32, 32))
        (32, 32)
        >>> fall_back_tuple((1, None), (32, 32))
        (1, 32)
        >>> fall_back_tuple(0, (32, 32))
        (32, 32)
        >>> fall_back_tuple(range(3), (32, 64, 48))
        (32, 1, 2)
        >>> fall_back_tuple([0], (32, 32))
        ValueError: sequence must have length 2, got length 1.

    """
    ndim = len(default)
    user = ensure_tuple_rep(user_provided, ndim)
    return tuple(  # use the default values if user provided is not valid
        user_c if func(user_c) else default_c for default_c, user_c in zip(default, user)
    )


def is_scalar_tensor(val) -> bool:
    if torch.is_tensor(val) and val.ndim == 0:
        return True
    return False


def is_scalar(val) -> bool:
    if torch.is_tensor(val) and val.ndim == 0:
        return True
    return bool(np.isscalar(val))


def progress_bar(index: int, count: int, desc: str = None, bar_len: int = 30, newline: bool = False) -> None:
    """print a progress bar to track some time consuming task.

    Args:
        index: current satus in progress.
        count: total steps of the progress.
        desc: description of the progress bar, if not None, show before the progress bar.
        bar_len: the total length of the bar on screen, default is 30 char.
        newline: whether to print in a new line for every index.
    """
    end = "\r" if newline is False else "\r\n"
    filled_len = int(bar_len * index // count)
    bar = f"{desc} " if desc is not None else ""
    bar += "[" + "=" * filled_len + " " * (bar_len - filled_len) + "]"
    print(f"{index}/{count} {bar}", end=end)
    if index == count:
        print("")


def get_seed():
    return _seed


def set_determinism(seed=np.iinfo(np.int32).max, additional_settings=None) -> None:
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

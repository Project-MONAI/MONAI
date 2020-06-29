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
from typing import Any, Tuple

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


def ensure_tuple(vals: Any) -> Tuple[Any, ...]:
    """
    Returns a tuple of `vals`
    """
    if not issequenceiterable(vals):
        vals = (vals,)

    return tuple(vals)


def ensure_tuple_size(tup, dim, pad_val=0):
    """
    Returns a copy of `tup` with `dim` values by either shortened or padded with `pad_val` as necessary.
    """
    tup = tuple(tup) + (pad_val,) * dim
    return tup[:dim]


def ensure_tuple_rep(tup, dim):
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


def adaptive_size(win_size, img_size):
    """
    Adapt `win_size` to `img_size`. Typically used when `win_size` is provided by the user,
    `img_size` is defined by data, this function returns an updated `win_size` with non-positive
    components replaced by the corresponding components from `img_size`.

    Examples::

        >>> adaptive_size(None, (32, 32))
        (32, 32)
        >>> adaptive_size((-1, 10), (32, 32))
        (32, 10)
        >>> adaptive_size((-1, None), (32, 32))
        (32, 32)
        >>> adaptive_size((1, None), (32, 32))
        (1, 32)
        >>> adaptive_size(0, (32, 32))
        (32, 32)
        >>> adaptive_size(range(3), (32, 64, 48))
        (32, 1, 2)
        >>> adaptive_size([0], (32, 32))
        ValueError: sequence must have length 2, got length 1.

    """
    ndim = len(img_size)
    w_size = ensure_tuple_rep(win_size, ndim)
    w_size = tuple(  # use the input image's size if spatial_size is not defined
        sp_d if (sp_d and sp_d > 0) else img_d for img_d, sp_d in zip(img_size, w_size)
    )
    return w_size


def is_scalar_tensor(val):
    if torch.is_tensor(val) and val.ndim == 0:
        return True
    return False


def is_scalar(val):
    if torch.is_tensor(val) and val.ndim == 0:
        return True
    return np.isscalar(val)


def progress_bar(index: int, count: int, desc: str = None, bar_len: int = 30, newline: bool = False):
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

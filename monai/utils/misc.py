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

import collections.abc
import itertools
import random
from ast import literal_eval
from distutils.util import strtobool
from typing import Any, Callable, Optional, Sequence, Tuple, Union

import numpy as np
import torch

__all__ = [
    "zip_with",
    "star_zip_with",
    "first",
    "issequenceiterable",
    "ensure_tuple",
    "ensure_tuple_size",
    "ensure_tuple_rep",
    "fall_back_tuple",
    "is_scalar_tensor",
    "is_scalar",
    "progress_bar",
    "get_seed",
    "set_determinism",
    "list_to_dict",
    "dtype_torch_to_numpy",
    "dtype_numpy_to_torch",
    "MAX_SEED",
]

_seed = None
_flag_deterministic = torch.backends.cudnn.deterministic
_flag_cudnn_benchmark = torch.backends.cudnn.benchmark
MAX_SEED = np.iinfo(np.uint32).max + 1  # 2**32, the actual seed should be in [0, MAX_SEED - 1] for uint32


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


def issequenceiterable(obj: Any) -> bool:
    """
    Determine if the object is an iterable sequence and is not a string.
    """
    if torch.is_tensor(obj):
        return int(obj.dim()) > 0  # a 0-d tensor is not iterable
    return isinstance(obj, collections.abc.Iterable) and not isinstance(obj, str)


def ensure_tuple(vals: Any) -> Tuple[Any, ...]:
    """
    Returns a tuple of `vals`.
    """
    if not issequenceiterable(vals):
        vals = (vals,)

    return tuple(vals)


def ensure_tuple_size(tup: Any, dim: int, pad_val: Any = 0) -> Tuple[Any, ...]:
    """
    Returns a copy of `tup` with `dim` values by either shortened or padded with `pad_val` as necessary.
    """
    tup = ensure_tuple(tup) + (pad_val,) * dim
    return tuple(tup[:dim])


def ensure_tuple_rep(tup: Any, dim: int) -> Tuple[Any, ...]:
    """
    Returns a copy of `tup` with `dim` values by either shortened or duplicated input.

    Raises:
        ValueError: When ``tup`` is a sequence and ``tup`` length is not ``dim``.

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
        ValueError: Sequence must have length 3, got length 2.

    """
    if not issequenceiterable(tup):
        return (tup,) * dim
    if len(tup) == dim:
        return tuple(tup)

    raise ValueError(f"Sequence must have length {dim}, got {len(tup)}.")


def fall_back_tuple(user_provided: Any, default: Sequence, func: Callable = lambda x: x and x > 0) -> Tuple[Any, ...]:
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
        ValueError: Sequence must have length 2, got length 1.

    """
    ndim = len(default)
    user = ensure_tuple_rep(user_provided, ndim)
    return tuple(  # use the default values if user provided is not valid
        user_c if func(user_c) else default_c for default_c, user_c in zip(default, user)
    )


def is_scalar_tensor(val: Any) -> bool:
    if torch.is_tensor(val) and val.ndim == 0:
        return True
    return False


def is_scalar(val: Any) -> bool:
    if torch.is_tensor(val) and val.ndim == 0:
        return True
    return bool(np.isscalar(val))


def progress_bar(index: int, count: int, desc: Optional[str] = None, bar_len: int = 30, newline: bool = False) -> None:
    """print a progress bar to track some time consuming task.

    Args:
        index: current status in progress.
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


def get_seed() -> Optional[int]:
    return _seed


def set_determinism(
    seed: Optional[int] = np.iinfo(np.uint32).max,
    additional_settings: Optional[Union[Sequence[Callable[[int], Any]], Callable[[int], Any]]] = None,
) -> None:
    """
    Set random seed for modules to enable or disable deterministic training.

    Args:
        seed: the random seed to use, default is np.iinfo(np.int32).max.
            It is recommended to set a large seed, i.e. a number that has a good balance
            of 0 and 1 bits. Avoid having many 0 bits in the seed.
            if set to None, will disable deterministic training.
        additional_settings: additional settings
            that need to set random seed.

    """
    if seed is None:
        # cast to 32 bit seed for CUDA
        seed_ = torch.default_generator.seed() % (np.iinfo(np.int32).max + 1)
        if not torch.cuda._is_in_bad_fork():
            torch.cuda.manual_seed_all(seed_)
    else:
        seed = int(seed) % MAX_SEED
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
    else:  # restore the original flags
        torch.backends.cudnn.deterministic = _flag_deterministic
        torch.backends.cudnn.benchmark = _flag_cudnn_benchmark


def list_to_dict(items):
    """
    To convert a list of "key=value" pairs into a dictionary.
    For examples: items: `["a=1", "b=2", "c=3"]`, return: {"a": "1", "b": "2", "c": "3"}.
    If no "=" in the pair, use None as the value, for example: ["a"], return: {"a": None}.
    Note that it will remove the blanks around keys and values.

    """

    def _parse_var(s):
        items = s.split("=", maxsplit=1)
        key = items[0].strip(" \n\r\t'")
        value = None
        if len(items) > 1:
            value = items[1].strip(" \n\r\t'")
        return key, value

    d = {}
    if items:
        for item in items:
            key, value = _parse_var(item)

            try:
                if key in d:
                    raise KeyError(f"encounter duplicated key {key}.")
                d[key] = literal_eval(value)
            except ValueError:
                try:
                    d[key] = bool(strtobool(str(value)))
                except ValueError:
                    d[key] = value
    return d


_torch_to_np_dtype = {
    torch.bool: np.bool,
    torch.uint8: np.uint8,
    torch.int8: np.int8,
    torch.int16: np.int16,
    torch.int32: np.int32,
    torch.int64: np.int64,
    torch.float16: np.float16,
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.complex64: np.complex64,
    torch.complex128: np.complex128,
}
_np_to_torch_dtype = {value: key for key, value in _torch_to_np_dtype.items()}


def dtype_torch_to_numpy(dtype):
    """Convert a torch dtype to its numpy equivalent."""
    return _torch_to_np_dtype[dtype]


def dtype_numpy_to_torch(dtype):
    """Convert a numpy dtype to its torch equivalent."""
    return _np_to_torch_dtype[dtype]

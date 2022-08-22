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

import inspect
import itertools
import os
import random
import shutil
import tempfile
import types
import warnings
from ast import literal_eval
from collections.abc import Iterable
from distutils.util import strtobool
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Tuple, Union, cast

import numpy as np
import torch

from monai.config.type_definitions import NdarrayOrTensor, NdarrayTensor, PathLike
from monai.utils.module import version_leq

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
    "MAX_SEED",
    "copy_to_device",
    "str2bool",
    "MONAIEnvVars",
    "ImageMetaKey",
    "is_module_ver_at_least",
    "has_option",
    "sample_slices",
    "check_parent_dir",
    "save_obj",
]

_seed = None
_flag_deterministic = torch.backends.cudnn.deterministic
_flag_cudnn_benchmark = torch.backends.cudnn.benchmark
NP_MAX = np.iinfo(np.uint32).max
MAX_SEED = NP_MAX + 1  # 2**32, the actual seed should be in [0, MAX_SEED - 1] for uint32


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
    try:
        if hasattr(obj, "ndim") and obj.ndim == 0:
            return False  # a 0-d tensor is not iterable
    except Exception:
        return False
    return isinstance(obj, Iterable) and not isinstance(obj, (str, bytes))


def ensure_tuple(vals: Any, wrap_array: bool = False) -> Tuple[Any, ...]:
    """
    Returns a tuple of `vals`.

    Args:
        vals: input data to convert to a tuple.
        wrap_array: if `True`, treat the input numerical array (ndarray/tensor) as one item of the tuple.
            if `False`, try to convert the array with `tuple(vals)`, default to `False`.

    """
    if wrap_array and isinstance(vals, (np.ndarray, torch.Tensor)):
        return (vals,)
    return tuple(vals) if issequenceiterable(vals) else (vals,)


def ensure_tuple_size(tup: Any, dim: int, pad_val: Any = 0) -> Tuple[Any, ...]:
    """
    Returns a copy of `tup` with `dim` values by either shortened or padded with `pad_val` as necessary.
    """
    new_tup = ensure_tuple(tup) + (pad_val,) * dim
    return new_tup[:dim]


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
    if isinstance(tup, torch.Tensor):
        tup = tup.detach().cpu().numpy()
    if isinstance(tup, np.ndarray):
        tup = tup.tolist()
    if not issequenceiterable(tup):
        return (tup,) * dim
    if len(tup) == dim:
        return tuple(tup)

    raise ValueError(f"Sequence must have length {dim}, got {len(tup)}.")


def fall_back_tuple(
    user_provided: Any, default: Union[Sequence, NdarrayTensor], func: Callable = lambda x: x and x > 0
) -> Tuple[Any, ...]:
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
    return isinstance(val, torch.Tensor) and val.ndim == 0


def is_scalar(val: Any) -> bool:
    if isinstance(val, torch.Tensor) and val.ndim == 0:
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
    end = "\r" if not newline else "\r\n"
    filled_len = int(bar_len * index // count)
    bar = f"{desc} " if desc is not None else ""
    bar += "[" + "=" * filled_len + " " * (bar_len - filled_len) + "]"
    print(f"{index}/{count} {bar}", end=end)
    if index == count:
        print("")


def get_seed() -> Optional[int]:
    return _seed


def set_determinism(
    seed: Optional[int] = NP_MAX,
    use_deterministic_algorithms: Optional[bool] = None,
    additional_settings: Optional[Union[Sequence[Callable[[int], Any]], Callable[[int], Any]]] = None,
) -> None:
    """
    Set random seed for modules to enable or disable deterministic training.

    Args:
        seed: the random seed to use, default is np.iinfo(np.int32).max.
            It is recommended to set a large seed, i.e. a number that has a good balance
            of 0 and 1 bits. Avoid having many 0 bits in the seed.
            if set to None, will disable deterministic training.
        use_deterministic_algorithms: Set whether PyTorch operations must use "deterministic" algorithms.
        additional_settings: additional settings that need to set random seed.

    Note:

        This function will not affect the randomizable objects in :py:class:`monai.transforms.Randomizable`, which
        have independent random states. For those objects, the ``set_random_state()`` method should be used to
        ensure the deterministic behavior (alternatively, :py:class:`monai.data.DataLoader` by default sets the seeds
        according to the global random state, please see also: :py:class:`monai.data.utils.worker_init_fn` and
        :py:class:`monai.data.utils.set_rnd`).
    """
    if seed is None:
        # cast to 32 bit seed for CUDA
        seed_ = torch.default_generator.seed() % MAX_SEED
        torch.manual_seed(seed_)
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

    if torch.backends.flags_frozen():
        warnings.warn("PyTorch global flag support of backends is disabled, enable it to set global `cudnn` flags.")
        torch.backends.__allow_nonbracketed_mutation_flag = True

    if seed is not None:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:  # restore the original flags
        torch.backends.cudnn.deterministic = _flag_deterministic
        torch.backends.cudnn.benchmark = _flag_cudnn_benchmark
    if use_deterministic_algorithms is not None:
        if hasattr(torch, "use_deterministic_algorithms"):  # `use_deterministic_algorithms` is new in torch 1.8.0
            torch.use_deterministic_algorithms(use_deterministic_algorithms)
        elif hasattr(torch, "set_deterministic"):  # `set_deterministic` is new in torch 1.7.0
            torch.set_deterministic(use_deterministic_algorithms)  # type: ignore
        else:
            warnings.warn("use_deterministic_algorithms=True, but PyTorch version is too old to set the mode.")


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
        value = items[1].strip(" \n\r\t'") if len(items) > 1 else None
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


def copy_to_device(
    obj: Any, device: Optional[Union[str, torch.device]], non_blocking: bool = True, verbose: bool = False
) -> Any:
    """
    Copy object or tuple/list/dictionary of objects to ``device``.

    Args:
        obj: object or tuple/list/dictionary of objects to move to ``device``.
        device: move ``obj`` to this device. Can be a string (e.g., ``cpu``, ``cuda``,
            ``cuda:0``, etc.) or of type ``torch.device``.
        non_blocking: when `True`, moves data to device asynchronously if
            possible, e.g., moving CPU Tensors with pinned memory to CUDA devices.
        verbose: when `True`, will print a warning for any elements of incompatible type
            not copied to ``device``.
    Returns:
        Same as input, copied to ``device`` where possible. Original input will be
            unchanged.
    """

    if hasattr(obj, "to"):
        return obj.to(device, non_blocking=non_blocking)
    if isinstance(obj, tuple):
        return tuple(copy_to_device(o, device, non_blocking) for o in obj)
    if isinstance(obj, list):
        return [copy_to_device(o, device, non_blocking) for o in obj]
    if isinstance(obj, dict):
        return {k: copy_to_device(o, device, non_blocking) for k, o in obj.items()}
    if verbose:
        fn_name = cast(types.FrameType, inspect.currentframe()).f_code.co_name
        warnings.warn(f"{fn_name} called with incompatible type: " + f"{type(obj)}. Data will be returned unchanged.")

    return obj


def str2bool(value: str, default: bool = False, raise_exc: bool = True) -> bool:
    """
    Convert a string to a boolean. Case insensitive.
    True: yes, true, t, y, 1. False: no, false, f, n, 0.

    Args:
        value: string to be converted to a boolean.
        raise_exc: if value not in tuples of expected true or false inputs,
            should we raise an exception? If not, return `None`.
    Raises
        ValueError: value not in tuples of expected true or false inputs and
            `raise_exc` is `True`.
    """
    true_set = ("yes", "true", "t", "y", "1")
    false_set = ("no", "false", "f", "n", "0")
    if isinstance(value, str):
        value = value.lower()
        if value in true_set:
            return True
        if value in false_set:
            return False

    if raise_exc:
        raise ValueError(f"Got \"{value}\", expected a value from: {', '.join(true_set + false_set)}")
    return default


class MONAIEnvVars:
    """
    Environment variables used by MONAI.
    """

    @staticmethod
    def data_dir() -> Optional[str]:
        return os.environ.get("MONAI_DATA_DIRECTORY", None)

    @staticmethod
    def debug() -> bool:
        val = os.environ.get("MONAI_DEBUG", False)
        return val if isinstance(val, bool) else str2bool(val)

    @staticmethod
    def doc_images() -> Optional[str]:
        return os.environ.get("MONAI_DOC_IMAGES", None)


class ImageMetaKey:
    """
    Common key names in the metadata header of images
    """

    FILENAME_OR_OBJ = "filename_or_obj"
    PATCH_INDEX = "patch_index"
    SPATIAL_SHAPE = "spatial_shape"


def has_option(obj, keywords: Union[str, Sequence[str]]) -> bool:
    """
    Return a boolean indicating whether the given callable `obj` has the `keywords` in its signature.
    """
    if not callable(obj):
        return False
    sig = inspect.signature(obj)
    return all(key in sig.parameters for key in ensure_tuple(keywords))


def is_module_ver_at_least(module, version):
    """Determine if a module's version is at least equal to the given value.

    Args:
        module: imported module's name, e.g., `np` or `torch`.
        version: required version, given as a tuple, e.g., `(1, 8, 0)`.
    Returns:
        `True` if module is the given version or newer.
    """
    test_ver = ".".join(map(str, version))
    return module.__version__ != test_ver and version_leq(test_ver, module.__version__)


def sample_slices(data: NdarrayOrTensor, dim: int = 1, as_indices: bool = True, *slicevals: int) -> NdarrayOrTensor:
    """sample several slices of input numpy array or Tensor on specified `dim`.

    Args:
        data: input data to sample slices, can be numpy array or PyTorch Tensor.
        dim: expected dimension index to sample slices, default to `1`.
        as_indices: if `True`, `slicevals` arg will be treated as the expected indices of slice, like: `1, 3, 5`
            means `data[..., [1, 3, 5], ...]`, if `False`, `slicevals` arg will be treated as args for `slice` func,
            like: `1, None` means `data[..., [1:], ...]`, `1, 5` means `data[..., [1: 5], ...]`.
        slicevals: indices of slices or start and end indices of expected slices, depends on `as_indices` flag.

    """
    slices = [slice(None)] * len(data.shape)
    slices[dim] = slicevals if as_indices else slice(*slicevals)  # type: ignore

    return data[tuple(slices)]


def check_parent_dir(path: PathLike, create_dir: bool = True):
    """
    Utility to check whether the parent directory of the `path` exists.

    Args:
        path: input path to check the parent directory.
        create_dir: if True, when the parent directory doesn't exist, create the directory,
            otherwise, raise exception.

    """
    path = Path(path)
    path_dir = path.parent
    if not path_dir.exists():
        if create_dir:
            path_dir.mkdir(parents=True)
        else:
            raise ValueError(f"the directory of specified path does not exist: `{path_dir}`.")


def save_obj(
    obj, path: PathLike, create_dir: bool = True, atomic: bool = True, func: Optional[Callable] = None, **kwargs
):
    """
    Save an object to file with specified path.
    Support to serialize to a temporary file first, then move to final destination,
    so that files are guaranteed to not be damaged if exception occurs.

    Args:
        obj: input object data to save.
        path: target file path to save the input object.
        create_dir: whether to create dictionary of the path if not existng, default to `True`.
        atomic: if `True`, state is serialized to a temporary file first, then move to final destination.
            so that files are guaranteed to not be damaged if exception occurs. default to `True`.
        func: the function to save file, if None, default to `torch.save`.
        kwargs: other args for the save `func` except for the checkpoint and filename.
            default `func` is `torch.save()`, details of other args:
            https://pytorch.org/docs/stable/generated/torch.save.html.

    """
    path = Path(path)
    check_parent_dir(path=path, create_dir=create_dir)
    if path.exists():
        # remove the existing file
        os.remove(path)

    if func is None:
        func = torch.save

    if not atomic:
        func(obj=obj, f=path, **kwargs)
        return
    try:
        # writing to a temporary directory and then using a nearly atomic rename operation
        with tempfile.TemporaryDirectory() as tempdir:
            temp_path: Path = Path(tempdir) / path.name
            func(obj=obj, f=temp_path, **kwargs)
            if temp_path.is_file():
                shutil.move(str(temp_path), path)
    except PermissionError:  # project-monai/monai issue #3613
        pass

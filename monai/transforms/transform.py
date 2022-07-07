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
A collection of generic interfaces for MONAI transforms.
"""

import logging
from abc import ABC, abstractmethod
from functools import wraps
from typing import Any, Callable, Dict, Generator, Hashable, Iterable, List, Mapping, Optional, Tuple, TypeVar, Union

import numpy as np
import torch

from monai import transforms
from monai.config import KeysCollection
from monai.data.meta_tensor import MetaTensor
from monai.utils import MAX_SEED, ensure_tuple, first
from monai.utils.enums import PostFix, TransformBackends

__all__ = ["ThreadUnsafe", "apply_transform", "Randomizable", "RandomizableTransform", "Transform", "MapTransform"]

ReturnType = TypeVar("ReturnType")


def _apply_transform(
    transform: Callable[..., ReturnType], parameters: Any, unpack_parameters: bool = False
) -> ReturnType:
    """
    Perform transformation `transform` with the provided parameters `parameters`.

    If `parameters` is a tuple and `unpack_items` is True, each parameter of `parameters` is unpacked
    as arguments to `transform`.
    Otherwise `parameters` is considered as single argument to `transform`.

    Args:
        transform: a callable to be used to transform `data`.
        parameters: parameters for the `transform`.
        unpack_parameters: whether to unpack parameters for `transform`. Defaults to False.

    Returns:
        ReturnType: The return type of `transform`.
    """
    if isinstance(parameters, tuple) and unpack_parameters:
        return transform(*parameters)

    return transform(parameters)


def apply_transform(
    transform: Callable[..., ReturnType],
    data: Any,
    map_items: bool = True,
    unpack_items: bool = False,
    log_stats: bool = False,
) -> Union[List[ReturnType], ReturnType]:
    """
    Transform `data` with `transform`.

    If `data` is a list or tuple and `map_data` is True, each item of `data` will be transformed
    and this method returns a list of outcomes.
    otherwise transform will be applied once with `data` as the argument.

    Args:
        transform: a callable to be used to transform `data`.
        data: an object to be transformed.
        map_items: whether to apply transform to each item in `data`,
            if `data` is a list or tuple. Defaults to True.
        unpack_items: whether to unpack parameters using `*`. Defaults to False.
        log_stats: whether to log the detailed information of data and applied transform when error happened,
            for NumPy array and PyTorch Tensor, log the data shape and value range,
            for other metadata, log the values directly. default to `False`.

    Raises:
        Exception: When ``transform`` raises an exception.

    Returns:
        Union[List[ReturnType], ReturnType]: The return type of `transform` or a list thereof.
    """
    try:
        if isinstance(data, (list, tuple)) and map_items:
            return [_apply_transform(transform, item, unpack_items) for item in data]
        return _apply_transform(transform, data, unpack_items)
    except Exception as e:

        if log_stats and not isinstance(transform, transforms.compose.Compose):
            # log the input data information of exact transform in the transform chain
            datastats = transforms.utility.array.DataStats(data_shape=False, value_range=False)
            logger = logging.getLogger(datastats._logger_name)
            logger.info(f"\n=== Transform input info -- {type(transform).__name__} ===")
            if isinstance(data, (list, tuple)):
                data = data[0]

            def _log_stats(data, prefix: Optional[str] = "Data"):
                if isinstance(data, (np.ndarray, torch.Tensor)):
                    # log data type, shape, range for array
                    datastats(img=data, data_shape=True, value_range=True, prefix=prefix)
                else:
                    # log data type and value for other metadata
                    datastats(img=data, data_value=True, prefix=prefix)

            if isinstance(data, dict):
                for k, v in data.items():
                    _log_stats(data=v, prefix=k)
            else:
                _log_stats(data=data)
        raise RuntimeError(f"applying transform {transform}") from e


class ThreadUnsafe:
    """
    A class to denote that the transform will mutate its member variables,
    when being applied. Transforms inheriting this class should be used
    cautiously in a multi-thread context.

    This type is typically used by :py:class:`monai.data.CacheDataset` and
    its extensions, where the transform cache is built with multiple threads.
    """

    pass


class Randomizable(ABC, ThreadUnsafe):
    """
    An interface for handling random state locally, currently based on a class
    variable `R`, which is an instance of `np.random.RandomState`.  This
    provides the flexibility of component-specific determinism without
    affecting the global states.  It is recommended to use this API with
    :py:class:`monai.data.DataLoader` for deterministic behaviour of the
    preprocessing pipelines. This API is not thread-safe. Additionally,
    deepcopying instance of this class often causes insufficient randomness as
    the random states will be duplicated.
    """

    R: np.random.RandomState = np.random.RandomState()

    def set_random_state(
        self, seed: Optional[int] = None, state: Optional[np.random.RandomState] = None
    ) -> "Randomizable":
        """
        Set the random state locally, to control the randomness, the derived
        classes should use :py:attr:`self.R` instead of `np.random` to introduce random
        factors.

        Args:
            seed: set the random state with an integer seed.
            state: set the random state with a `np.random.RandomState` object.

        Raises:
            TypeError: When ``state`` is not an ``Optional[np.random.RandomState]``.

        Returns:
            a Randomizable instance.

        """
        if seed is not None:
            _seed = id(seed) if not isinstance(seed, (int, np.integer)) else seed
            _seed = _seed % MAX_SEED
            self.R = np.random.RandomState(_seed)
            return self

        if state is not None:
            if not isinstance(state, np.random.RandomState):
                raise TypeError(f"state must be None or a np.random.RandomState but is {type(state).__name__}.")
            self.R = state
            return self

        self.R = np.random.RandomState()
        return self

    def randomize(self, data: Any) -> None:
        """
        Within this method, :py:attr:`self.R` should be used, instead of `np.random`, to introduce random factors.

        all :py:attr:`self.R` calls happen here so that we have a better chance to
        identify errors of sync the random state.

        This method can generate the random factors based on properties of the input data.

        Raises:
            NotImplementedError: When the subclass does not override this method.
        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")


class Transform(ABC):
    """
    An abstract class of a ``Transform``.
    A transform is callable that processes ``data``.

    It could be stateful and may modify ``data`` in place,
    the implementation should be aware of:

        #. thread safety when mutating its own states.
           When used from a multi-process context, transform's instance variables are read-only.
           thread-unsafe transforms should inherit :py:class:`monai.transforms.ThreadUnsafe`.
        #. ``data`` content unused by this transform may still be used in the
           subsequent transforms in a composed transform.
        #. storing too much information in ``data`` may cause some memory issue or IPC sync issue,
           especially in the multi-processing environment of PyTorch DataLoader.

    See Also

        :py:class:`monai.transforms.Compose`
    """

    # Transforms should add data types to this list if they are capable of performing a transform without
    # modifying the input type. For example, ["torch.Tensor", "np.ndarray"] means that no copies of the data
    # are required if the input is either `torch.Tensor` or `np.ndarray`.
    backend: List[TransformBackends] = []

    @abstractmethod
    def __call__(self, data: Any):
        """
        ``data`` is an element which often comes from an iteration over an
        iterable, such as :py:class:`torch.utils.data.Dataset`. This method should
        return an updated version of ``data``.
        To simplify the input validations, most of the transforms assume that

        - ``data`` is a Numpy ndarray, PyTorch Tensor or string,
        - the data shape can be:

          #. string data without shape, `LoadImage` transform expects file paths,
          #. most of the pre-/post-processing transforms expect: ``(num_channels, spatial_dim_1[, spatial_dim_2, ...])``,
             except for example: `AddChannel` expects (spatial_dim_1[, spatial_dim_2, ...]) and
             `AsChannelFirst` expects (spatial_dim_1[, spatial_dim_2, ...], num_channels),

        - the channel dimension is often not omitted even if number of channels is one.

        This method can optionally take additional arguments to help execute transformation operation.

        Raises:
            NotImplementedError: When the subclass does not override this method.

        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")


class RandomizableTransform(Randomizable, Transform):
    """
    An interface for handling random state locally, currently based on a class variable `R`,
    which is an instance of `np.random.RandomState`.
    This class introduces a randomized flag `_do_transform`, is mainly for randomized data augmentation transforms.
    For example:

    .. code-block:: python

        from monai.transforms import RandomizableTransform

        class RandShiftIntensity100(RandomizableTransform):
            def randomize(self):
                super().randomize(None)
                self._offset = self.R.uniform(low=0, high=100)

            def __call__(self, img):
                self.randomize()
                if not self._do_transform:
                    return img
                return img + self._offset

        transform = RandShiftIntensity()
        transform.set_random_state(seed=0)
        print(transform(10))

    """

    def __init__(self, prob: float = 1.0, do_transform: bool = True):
        self._do_transform = do_transform
        self.prob = min(max(prob, 0.0), 1.0)

    def randomize(self, data: Any) -> None:
        """
        Within this method, :py:attr:`self.R` should be used, instead of `np.random`, to introduce random factors.

        all :py:attr:`self.R` calls happen here so that we have a better chance to
        identify errors of sync the random state.

        This method can generate the random factors based on properties of the input data.
        """
        self._do_transform = self.R.rand() < self.prob


def attach_post_hook(func, hook):
    """adds `hook` after `func` calls using `func`'s return value"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        out = func(*args, **kwargs)
        return hook(args[0], out)

    return wrapper


class MapTransform(Transform):
    """
    A subclass of :py:class:`monai.transforms.Transform` with an assumption
    that the ``data`` input of ``self.__call__`` is a MutableMapping such as ``dict``.

    The ``keys`` parameter will be used to get and set the actual data
    item to transform.  That is, the callable of this transform should
    follow the pattern:

        .. code-block:: python

            def __call__(self, data):
                for key in self.keys:
                    if key in data:
                        # update output data with some_transform_function(data[key]).
                    else:
                        # raise exception unless allow_missing_keys==True.
                return data

    Raises:
        ValueError: When ``keys`` is an empty iterable.
        TypeError: When ``keys`` type is not in ``Union[Hashable, Iterable[Hashable]]``.

    """

    def __new__(cls, *args, **kwargs):
        cls.__call__ = attach_post_hook(cls.__call__, MapTransform.comp_update)
        return super().__new__(cls)

    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False) -> None:
        self.keys: Tuple[Hashable, ...] = ensure_tuple(keys)
        self.allow_missing_keys = allow_missing_keys
        if not self.keys:
            raise ValueError("keys must be non empty.")
        for key in self.keys:
            if not isinstance(key, Hashable):
                raise TypeError(f"keys must be one of (Hashable, Iterable[Hashable]) but is {type(keys).__name__}.")

    def comp_update(self, data):
        """
        This function is to be called after every `self.__call__(data)`,
        update `data[key_transforms]` and `data[key_meta_dict]` using the content from MetaTensor `data[key]`,
        for MetaTensor backward compatibility.
        """
        if not isinstance(data, Mapping):
            return data
        d = dict(data)
        for k in data:
            if not isinstance(data[k], MetaTensor):
                continue
            meta_dict_key = PostFix.meta(k)
            if meta_dict_key in d:
                d[meta_dict_key].update(data[k].meta)
            if data[k].applied_operations:
                transform_key = transforms.TraceableTransform.trace_key(k)
                d[transform_key] = data[k].applied_operations
        return d

    @abstractmethod
    def __call__(self, data):
        """
        ``data`` often comes from an iteration over an iterable,
        such as :py:class:`torch.utils.data.Dataset`.

        To simplify the input validations, this method assumes:

        - ``data`` is a Python dictionary,
        - ``data[key]`` is a Numpy ndarray, PyTorch Tensor or string, where ``key`` is an element
          of ``self.keys``, the data shape can be:

          #. string data without shape, `LoadImaged` transform expects file paths,
          #. most of the pre-/post-processing transforms expect: ``(num_channels, spatial_dim_1[, spatial_dim_2, ...])``,
             except for example: `AddChanneld` expects (spatial_dim_1[, spatial_dim_2, ...]) and
             `AsChannelFirstd` expects (spatial_dim_1[, spatial_dim_2, ...], num_channels)

        - the channel dimension is often not omitted even if number of channels is one.

        Raises:
            NotImplementedError: When the subclass does not override this method.

        returns:
            An updated dictionary version of ``data`` by applying the transform.

        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")

    def key_iterator(self, data: Mapping[Hashable, Any], *extra_iterables: Optional[Iterable]) -> Generator:
        """
        Iterate across keys and optionally extra iterables. If key is missing, exception is raised if
        `allow_missing_keys==False` (default). If `allow_missing_keys==True`, key is skipped.

        Args:
            data: data that the transform will be applied to
            extra_iterables: anything else to be iterated through
        """
        # if no extra iterables given, create a dummy list of Nones
        ex_iters = extra_iterables or [[None] * len(self.keys)]

        # loop over keys and any extra iterables
        _ex_iters: List[Any]
        for key, *_ex_iters in zip(self.keys, *ex_iters):
            # all normal, yield (what we yield depends on whether extra iterables were given)
            if key in data:
                yield (key,) + tuple(_ex_iters) if extra_iterables else key
            elif not self.allow_missing_keys:
                raise KeyError(
                    f"Key `{key}` of transform `{self.__class__.__name__}` was missing in the data"
                    " and allow_missing_keys==False."
                )

    def first_key(self, data: Dict[Hashable, Any]):
        """
        Get the first available key of `self.keys` in the input `data` dictionary.
        If no available key, return an empty list `[]`.

        Args:
            data: data that the transform will be applied to.

        """
        return first(self.key_iterator(data), [])

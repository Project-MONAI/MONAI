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

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

import monai
import monai.transforms as mt
from monai.apps.utils import get_logger
from monai.transforms.inverse import InvertibleTransform

# For backwards compatibility (so this still works: from monai.transforms.compose import MapTransform)
from monai.transforms.transform import (  # noqa: F401
    LazyTransform,
    MapTransform,
    Randomizable,
    RandomizableTransform,
    Transform,
    apply_transform,
)
from monai.utils import MAX_SEED, TraceKeys, ensure_tuple, get_seed
from monai.utils.misc import to_tuple_of_dictionaries

logger = get_logger(__name__)

__all__ = ["Compose", "OneOf", "RandomOrder", "evaluate_with_overrides"]


def evaluate_with_overrides(
    data,
    upcoming,
    lazy_evaluation: bool | None = False,
    overrides: dict | None = None,
    override_keys: Sequence[str] | None = None,
    verbose: bool = False,
):
    """
    The previously applied transform may have been lazily applied to MetaTensor `data` and
    made `data.has_pending_operations` equals to True. Given the upcoming transform ``upcoming``,
    this function determines whether `data.pending_operations` should be evaluated. If so, it will
    evaluate the lazily applied transforms.

    Currently, the conditions for evaluation are:

        - ``lazy_evaluation`` is ``True``, AND
        - the data is a ``MetaTensor`` and has pending operations, AND
        - the upcoming transform is an instance of ``Identity`` or ``IdentityD`` or ``None``.

    The returned `data` will then be ready for the ``upcoming`` transform.

    Args:
        data: data to be evaluated.
        upcoming: the upcoming transform.
        lazy_evaluation: whether to evaluate the pending operations.
        override: keyword arguments to apply transforms.
        override_keys: to which the override arguments are used when apply transforms.
        verbose: whether to print debugging info when evaluate MetaTensor with pending operations.

    """
    if not lazy_evaluation:
        return data  # eager evaluation
    overrides = (overrides or {}).copy()
    if isinstance(data, monai.data.MetaTensor):
        if data.has_pending_operations and (
            (upcoming is None)  # final transform
            or (isinstance(upcoming, mt.Identity))  # Identity -> eval all keys
            or (isinstance(upcoming, mt.Identityd) and override_keys in upcoming.keys)
        ):
            data, _ = mt.apply_transforms(data, None, overrides=overrides)
            if verbose:
                next_name = "final output" if upcoming is None else f"'{upcoming.__class__.__name__}'"
                logger.info(f"Evaluated - '{override_keys}' - up-to-date for - {next_name}")
        elif verbose:
            logger.info(
                f"Lazy - '{override_keys}' - upcoming: '{upcoming.__class__.__name__}'"
                f"- pending {len(data.pending_operations)}"
            )
        return data
    override_keys = ensure_tuple(override_keys)
    if isinstance(data, dict):
        if isinstance(upcoming, MapTransform):
            applied_keys = {k for k in data if k in upcoming.keys}
            if not applied_keys:
                return data
        else:
            applied_keys = set(data.keys())

        keys_to_override = {k for k in applied_keys if k in override_keys}
        # generate a list of dictionaries with the appropriate override value per key
        dict_overrides = to_tuple_of_dictionaries(overrides, override_keys)
        for k in data:
            if k in keys_to_override:
                dict_for_key = dict_overrides[override_keys.index(k)]
                data[k] = evaluate_with_overrides(data[k], upcoming, lazy_evaluation, dict_for_key, k, verbose)
            else:
                data[k] = evaluate_with_overrides(data[k], upcoming, lazy_evaluation, None, k, verbose)

    if isinstance(data, (list, tuple)):
        return [evaluate_with_overrides(v, upcoming, lazy_evaluation, overrides, override_keys, verbose) for v in data]
    return data


class Compose(Randomizable, InvertibleTransform):
    """
    ``Compose`` provides the ability to chain a series of callables together in
    a sequential manner. Each transform in the sequence must take a single
    argument and return a single value.

    ``Compose`` can be used in two ways:

    #. With a series of transforms that accept and return a single
       ndarray / tensor / tensor-like parameter.
    #. With a series of transforms that accept and return a dictionary that
       contains one or more parameters. Such transforms must have pass-through
       semantics that unused values in the dictionary must be copied to the return
       dictionary. It is required that the dictionary is copied between input
       and output of each transform.

    If some transform takes a data item dictionary as input, and returns a
    sequence of data items in the transform chain, all following transforms
    will be applied to each item of this list if `map_items` is `True` (the
    default).  If `map_items` is `False`, the returned sequence is passed whole
    to the next callable in the chain.

    For example:

    A `Compose([transformA, transformB, transformC],
    map_items=True)(data_dict)` could achieve the following patch-based
    transformation on the `data_dict` input:

    #. transformA normalizes the intensity of 'img' field in the `data_dict`.
    #. transformB crops out image patches from the 'img' and 'seg' of
       `data_dict`, and return a list of three patch samples::

        {'img': 3x100x100 data, 'seg': 1x100x100 data, 'shape': (100, 100)}
                             applying transformB
                                 ---------->
        [{'img': 3x20x20 data, 'seg': 1x20x20 data, 'shape': (20, 20)},
         {'img': 3x20x20 data, 'seg': 1x20x20 data, 'shape': (20, 20)},
         {'img': 3x20x20 data, 'seg': 1x20x20 data, 'shape': (20, 20)},]

    #. transformC then randomly rotates or flips 'img' and 'seg' of
       each dictionary item in the list returned by transformB.

    The composed transforms will be set the same global random seed if user called
    `set_determinism()`.

    When using the pass-through dictionary operation, you can make use of
    :class:`monai.transforms.adaptors.adaptor` to wrap transforms that don't conform
    to the requirements. This approach allows you to use transforms from
    otherwise incompatible libraries with minimal additional work.

    Note:

        In many cases, Compose is not the best way to create pre-processing
        pipelines. Pre-processing is often not a strictly sequential series of
        operations, and much of the complexity arises when a not-sequential
        set of functions must be called as if it were a sequence.

        Example: images and labels
        Images typically require some kind of normalization that labels do not.
        Both are then typically augmented through the use of random rotations,
        flips, and deformations.
        Compose can be used with a series of transforms that take a dictionary
        that contains 'image' and 'label' entries. This might require wrapping
        `torchvision` transforms before passing them to compose.
        Alternatively, one can create a class with a `__call__` function that
        calls your pre-processing functions taking into account that not all of
        them are called on the labels.

    Args:
        transforms: sequence of callables.
        map_items: whether to apply transform to each item in the input `data` if `data` is a list or tuple.
            defaults to `True`.
        unpack_items: whether to unpack input `data` with `*` as parameters for the callable function of transform.
            defaults to `False`.
        log_stats: whether to log the detailed information of data and applied transform when error happened,
            for NumPy array and PyTorch Tensor, log the data shape and value range,
            for other metadata, log the values directly. default to `False`.
        lazy_evaluation: whether to enable lazy evaluation for lazy transforms. If False, transforms will be
            carried out on a transform by transform basis. If True, all lazy transforms will
            be executed by accumulating changes and resampling as few times as possible.
            A `monai.transforms.Identity[D]` transform in the pipeline will trigger the evaluation of
            the pending operations and make the primary data up-to-date.
        overrides: this optional parameter allows you to specify a dictionary of parameters that should be overridden
            when executing a pipeline. These each parameter that is compatible with a given transform is then applied
            to that transform before it is executed. Note that overrides are currently only applied when lazy_evaluation
            is True. If lazy_evaluation is False they are ignored.
            currently supported args are:
            {``"mode"``, ``"padding_mode"``, ``"dtype"``, ``"align_corners"``, ``"resample_mode"``, ``device``},
            please see also :py:func:`monai.transforms.lazy.apply_transforms` for more details.
        override_keys: this optional parameter specifies the keys to which ``overrides`` are to be applied. If
            ``overrides`` is set, ``override_keys`` must also be set.
        verbose: whether to print debugging info when lazy_evaluation=True.
    """

    def __init__(
        self,
        transforms: Sequence[Callable] | Callable | None = None,
        map_items: bool = True,
        unpack_items: bool = False,
        log_stats: bool = False,
        lazy_evaluation: bool | None = None,
        overrides: dict | None = None,
        override_keys: Sequence[str] | None = None,
        verbose: bool = False,
    ) -> None:
        if transforms is None:
            transforms = []
        self.transforms = ensure_tuple(transforms)
        self.map_items = map_items
        self.unpack_items = unpack_items
        self.log_stats = log_stats
        self.set_random_state(seed=get_seed())

        self.lazy_evaluation = lazy_evaluation
        self.overrides = overrides
        self.override_keys = override_keys
        self.verbose = verbose

        if self.lazy_evaluation is not None:
            for t in self.flatten().transforms:  # TODO: test Compose of Compose/OneOf
                if isinstance(t, LazyTransform):
                    t.lazy_evaluation = self.lazy_evaluation

    def set_random_state(self, seed: int | None = None, state: np.random.RandomState | None = None) -> Compose:
        super().set_random_state(seed=seed, state=state)
        for _transform in self.transforms:
            if not isinstance(_transform, Randomizable):
                continue
            _transform.set_random_state(seed=self.R.randint(MAX_SEED, dtype="uint32"))
        return self

    def randomize(self, data: Any | None = None) -> None:
        for _transform in self.transforms:
            if not isinstance(_transform, Randomizable):
                continue
            try:
                _transform.randomize(data)
            except TypeError as type_error:
                tfm_name: str = type(_transform).__name__
                warnings.warn(
                    f'Transform "{tfm_name}" in Compose not randomized\n{tfm_name}.{type_error}.', RuntimeWarning
                )

    def flatten(self):
        """Return a Composition with a simple list of transforms, as opposed to any nested Compositions.

        e.g., `t1 = Compose([x, x, x, x, Compose([Compose([x, x]), x, x])]).flatten()`
        will result in the equivalent of `t1 = Compose([x, x, x, x, x, x, x, x])`.

        """
        new_transforms = []
        for t in self.transforms:
            if type(t) is Compose:  # nopep8
                new_transforms += t.flatten().transforms
            else:
                new_transforms.append(t)

        return Compose(new_transforms)

    def __len__(self):
        """Return number of transformations."""
        return len(self.flatten().transforms)

    def evaluate_with_overrides(self, input_, upcoming_xform):
        """
        Args:
            input_: input data to be transformed.
            upcoming_xform: a transform used to determine whether to evaluate with override
        """
        return evaluate_with_overrides(
            input_,
            upcoming_xform,
            lazy_evaluation=self.lazy_evaluation,
            overrides=self.overrides,
            override_keys=self.override_keys,
            verbose=self.verbose,
        )

    def __call__(self, input_):
        for _transform in self.transforms:
            input_ = self.evaluate_with_overrides(input_, _transform)
            input_ = apply_transform(_transform, input_, self.map_items, self.unpack_items, self.log_stats)
        input_ = self.evaluate_with_overrides(input_, None)
        return input_

    def inverse(self, data):
        invertible_transforms = [t for t in self.flatten().transforms if isinstance(t, InvertibleTransform)]
        if not invertible_transforms:
            warnings.warn("inverse has been called but no invertible transforms have been supplied")

        # loop backwards over transforms
        for t in reversed(invertible_transforms):
            data = apply_transform(t.inverse, data, self.map_items, self.unpack_items, self.log_stats)
        return data


class OneOf(Compose):
    """
    ``OneOf`` provides the ability to randomly choose one transform out of a
    list of callables with pre-defined probabilities for each.

    Args:
        transforms: sequence of callables.
        weights: probabilities corresponding to each callable in transforms.
            Probabilities are normalized to sum to one.
        map_items: whether to apply transform to each item in the input `data` if `data` is a list or tuple.
            defaults to `True`.
        unpack_items: whether to unpack input `data` with `*` as parameters for the callable function of transform.
            defaults to `False`.
        log_stats: whether to log the detailed information of data and applied transform when error happened,
            for NumPy array and PyTorch Tensor, log the data shape and value range,
            for other metadata, log the values directly. default to `False`.
        lazy_evaluation: whether to enable lazy evaluation for lazy transforms. If True, all lazy transforms will
            be executed by accumulating changes and resampling as few times as possible. If False, transforms will be
            carried out on a transform by transform basis.
            A `monai.transforms.Identity[D]` transform in the pipeline will trigger the evaluation of
            the pending operations and make the primary data up-to-date.
        overrides: this optional parameter allows you to specify a dictionary of parameters that should be overridden
            when executing a pipeline. These each parameter that is compatible with a given transform is then applied
            to that transform before it is executed. Note that overrides are currently only applied when lazy_evaluation
            is True. If lazy_evaluation is False they are ignored.
            currently supported args are:
            {``"mode"``, ``"padding_mode"``, ``"dtype"``, ``"align_corners"``, ``"resample_mode"``, ``device``},
            please see also :py:func:`monai.transforms.lazy.apply_transforms` for more details.
        override_keys: this optional parameter specifies the keys to which ``overrides`` are to be applied. If
            ``overrides`` is set, ``override_keys`` must also be set.
        verbose: whether to print debugging info when lazy_evaluation=True.
    """

    def __init__(
        self,
        transforms: Sequence[Callable] | Callable | None = None,
        weights: Sequence[float] | float | None = None,
        map_items: bool = True,
        unpack_items: bool = False,
        log_stats: bool = False,
        lazy_evaluation: bool | None = None,
        overrides: dict | None = None,
        override_keys: Sequence[str] | None = None,
        verbose: bool = False,
    ) -> None:
        super().__init__(
            transforms, map_items, unpack_items, log_stats, lazy_evaluation, overrides, override_keys, verbose
        )
        if len(self.transforms) == 0:
            weights = []
        elif weights is None or isinstance(weights, float):
            weights = [1.0 / len(self.transforms)] * len(self.transforms)
        if len(weights) != len(self.transforms):
            raise ValueError(
                "transforms and weights should be same size if both specified as sequences, "
                f"got {len(weights)} and {len(self.transforms)}."
            )
        self.weights = ensure_tuple(self._normalize_probabilities(weights))

    def _normalize_probabilities(self, weights):
        if len(weights) == 0:
            return weights
        weights = np.array(weights)
        if np.any(weights < 0):
            raise ValueError(f"Probabilities must be greater than or equal to zero, got {weights}.")
        if np.all(weights == 0):
            raise ValueError(f"At least one probability must be greater than zero, got {weights}.")
        weights = weights / weights.sum()
        return list(weights)

    def flatten(self):
        transforms = []
        weights = []
        for t, w in zip(self.transforms, self.weights):
            # if nested, probability is the current weight multiplied by the nested weights,
            # and so on recursively
            if isinstance(t, OneOf):
                tr = t.flatten()
                for t_, w_ in zip(tr.transforms, tr.weights):
                    transforms.append(t_)
                    weights.append(w_ * w)
            else:
                transforms.append(t)
                weights.append(w)
        return OneOf(transforms, weights, self.map_items, self.unpack_items)

    def __call__(self, data):
        if len(self.transforms) == 0:
            return data
        index = self.R.multinomial(1, self.weights).argmax()
        _transform = self.transforms[index]
        data = apply_transform(_transform, data, self.map_items, self.unpack_items, self.log_stats)
        # if the data is a mapping (dictionary), append the OneOf transform to the end
        if isinstance(data, monai.data.MetaTensor):
            self.push_transform(data, extra_info={"index": index})
        elif isinstance(data, Mapping):
            for key in data:  # dictionary not change size during iteration
                if isinstance(data[key], monai.data.MetaTensor):
                    self.push_transform(data[key], extra_info={"index": index})
        return data

    def inverse(self, data):
        if len(self.transforms) == 0:
            return data

        index = None
        if isinstance(data, monai.data.MetaTensor):
            index = self.pop_transform(data)[TraceKeys.EXTRA_INFO]["index"]
        elif isinstance(data, Mapping):
            for key in data:
                if isinstance(data[key], monai.data.MetaTensor):
                    index = self.pop_transform(data, key)[TraceKeys.EXTRA_INFO]["index"]
        else:
            raise RuntimeError(
                f"Inverse only implemented for Mapping (dictionary) or MetaTensor data, got type {type(data)}."
            )
        if index is None:
            # no invertible transforms have been applied
            return data

        _transform = self.transforms[index]
        # apply the inverse
        return _transform.inverse(data) if isinstance(_transform, InvertibleTransform) else data


class RandomOrder(Compose):
    """
    ``RandomOrder`` provides the ability to apply a list of transformations in random order.

    Args:
        transforms: sequence of callables.
        map_items: whether to apply transform to each item in the input `data` if `data` is a list or tuple.
            defaults to `True`.
        unpack_items: whether to unpack input `data` with `*` as parameters for the callable function of transform.
            defaults to `False`.
        log_stats: whether to log the detailed information of data and applied transform when error happened,
            for NumPy array and PyTorch Tensor, log the data shape and value range,
            for other metadata, log the values directly. default to `False`.
        lazy_evaluation: whether to enable lazy evaluation for lazy transforms. If True, all lazy transforms will
            be executed by accumulating changes and resampling as few times as possible. If False, transforms will be
            carried out on a transform by transform basis.
            A `monai.transforms.Identity[D]` transform in the pipeline will trigger the evaluation of
            the pending operations and make the primary data up-to-date.
        overrides: this optional parameter allows you to specify a dictionary of parameters that should be overridden
            when executing a pipeline. These each parameter that is compatible with a given transform is then applied
            to that transform before it is executed. Note that overrides are currently only applied when lazy_evaluation
            is True. If lazy_evaluation is False they are ignored.
            currently supported args are:
            {``"mode"``, ``"padding_mode"``, ``"dtype"``, ``"align_corners"``, ``"resample_mode"``, ``device``},
            please see also :py:func:`monai.transforms.lazy.apply_transforms` for more details.
        override_keys: this optional parameter specifies the keys to which ``overrides`` are to be applied. If
            ``overrides`` is set, ``override_keys`` must also be set.
        verbose: whether to print debugging info when lazy_evaluation=True.
    """

    def __init__(
        self,
        transforms: Sequence[Callable] | Callable | None = None,
        map_items: bool = True,
        unpack_items: bool = False,
        log_stats: bool = False,
        lazy_evaluation: bool | None = None,
        overrides: dict | None = None,
        override_keys: Sequence[str] | None = None,
        verbose: bool = False,
    ) -> None:
        super().__init__(
            transforms, map_items, unpack_items, log_stats, lazy_evaluation, overrides, override_keys, verbose
        )

    def __call__(self, input_):
        if len(self.transforms) == 0:
            return input_
        num = len(self.transforms)
        applied_order = self.R.permutation(range(num))

        for index in applied_order:
            input_ = apply_transform(self.transforms[index], input_, self.map_items, self.unpack_items, self.log_stats)
        # if the data is a mapping (dictionary), append the RandomOrder transform to the end
        if isinstance(input_, monai.data.MetaTensor):
            self.push_transform(input_, extra_info={"applied_order": applied_order})
        elif isinstance(input_, Mapping):
            for key in input_:  # dictionary not change size during iteration
                if isinstance(input_[key], monai.data.MetaTensor):
                    self.push_transform(input_[key], extra_info={"applied_order": applied_order})
        return input_

    def inverse(self, data):
        if len(self.transforms) == 0:
            return data

        applied_order = None
        if isinstance(data, monai.data.MetaTensor):
            applied_order = self.pop_transform(data)[TraceKeys.EXTRA_INFO]["applied_order"]
        elif isinstance(data, Mapping):
            for key in data:
                if isinstance(data[key], monai.data.MetaTensor):
                    applied_order = self.pop_transform(data, key)[TraceKeys.EXTRA_INFO]["applied_order"]
        else:
            raise RuntimeError(
                f"Inverse only implemented for Mapping (dictionary) or MetaTensor data, got type {type(data)}."
            )
        if applied_order is None:
            # no invertible transforms have been applied
            return data

        # loop backwards over transforms
        for o in reversed(applied_order):
            if isinstance(self.transforms[o], InvertibleTransform):
                data = apply_transform(
                    self.transforms[o].inverse, data, self.map_items, self.unpack_items, self.log_stats
                )
        return data

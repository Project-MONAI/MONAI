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
A collection of generic interfaces for MONAI transforms.
"""

import warnings
from typing import Any, Callable, Optional, Sequence, Union

import numpy as np

from monai.transforms.inverse import InvertibleTransform

# For backwards compatibility (so this still works: from monai.transforms.compose import MapTransform)
from monai.transforms.transform import (  # noqa: F401
    MapTransform,
    Randomizable,
    RandomizableTransform,
    Transform,
    apply_transform,
)
from monai.utils import MAX_SEED, ensure_tuple, get_seed

__all__ = ["Compose"]


class Compose(Randomizable, InvertibleTransform):
    """
    ``Compose`` provides the ability to chain a series of calls together in a
    sequence. Each transform in the sequence must take a single argument and
    return a single value, so that the transforms can be called in a chain.

    ``Compose`` can be used in two ways:

    #. With a series of transforms that accept and return a single
       ndarray / tensor / tensor-like parameter.
    #. With a series of transforms that accept and return a dictionary that
       contains one or more parameters. Such transforms must have pass-through
       semantics; unused values in the dictionary must be copied to the return
       dictionary. It is required that the dictionary is copied between input
       and output of each transform.

    If some transform generates a list batch of data in the transform chain,
    every item in the list is still a dictionary, and all the following
    transforms will apply to every item of the list, for example:

    #. transformA normalizes the intensity of 'img' field in the dict data.
    #. transformB crops out a list batch of images on 'img' and 'seg' field.
       And constructs a list of dict data, other fields are copied::

            {                          [{                   {
                'img': [1, 2],              'img': [1],         'img': [2],
                'seg': [1, 2],              'seg': [1],         'seg': [2],
                'extra': 123,    -->        'extra': 123,       'extra': 123,
                'shape': 'CHWD'             'shape': 'CHWD'     'shape': 'CHWD'
            }                           },                  }]

    #. transformC then randomly rotates or flips 'img' and 'seg' fields of
       every dictionary item in the list.

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
    """

    def __init__(self, transforms: Optional[Union[Sequence[Callable], Callable]] = None) -> None:
        if transforms is None:
            transforms = []
        self.transforms = ensure_tuple(transforms)
        self.set_random_state(seed=get_seed())

    def set_random_state(self, seed: Optional[int] = None, state: Optional[np.random.RandomState] = None) -> "Compose":
        super().set_random_state(seed=seed, state=state)
        for _transform in self.transforms:
            if not isinstance(_transform, Randomizable):
                continue
            _transform.set_random_state(seed=self.R.randint(MAX_SEED, dtype="uint32"))
        return self

    def randomize(self, data: Optional[Any] = None) -> None:
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
            if isinstance(t, Compose):
                new_transforms += t.flatten().transforms
            else:
                new_transforms.append(t)

        return Compose(new_transforms)

    def __len__(self):
        """Return number of transformations."""
        return len(self.flatten().transforms)

    def __call__(self, input_):
        for _transform in self.transforms:
            input_ = apply_transform(_transform, input_)
        return input_

    def inverse(self, data):
        invertible_transforms = [t for t in self.flatten().transforms if isinstance(t, InvertibleTransform)]
        if len(invertible_transforms) == 0:
            warnings.warn("inverse has been called but no invertible transforms have been supplied")

        # loop backwards over transforms
        for t in reversed(invertible_transforms):
            data = apply_transform(t.inverse, data)
        return data

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

import numpy as np


class Transform:
    """
    An abstract class of a ``Transform``.
    A transform is callable that processes ``data``.

    It could be stateful and may modify ``data`` in place,
    the implementation should be aware of:
    - thread safety when mutating its own states.
        When used from a multi-process context, transform's instance variables are read-only.
    - ``data`` content unused by this transform may still be used in the
        subsequent transforms in a composed transform.
        see also: `monai.transforms.compose.Compose`.
    - storing too much information in ``data`` may not scale.
    """

    def __call__(self, data):
        """
        ``data`` is an element which often comes from an iteration over an
        iterable, such as``torch.utils.data.Dataset``. This method should
        return an updated version of ``data``.
        """
        raise NotImplementedError


class Compose:
    """
    `Compose` provides the ability to chain a series of calls together in a
    sequence. Each transform in the sequence must take a single argument and
    return a single value, so that the transforms can be called in a chain.

    `Compose` can be used in two ways:
    1. With a series of transforms that accept and return a single ndarray /
    / tensor / tensor-like parameter
    2. With a series of transforms that accept and return a dictionary that
    contains one or more parameters. Such transforms must have pass-through
    semantics; unused values in the dictionary must be copied to the return
    dictionary. It is required that the dictionary is copied between input
    and output of each transform.

    When using the pass-through dictionary operation, you can make use of
    `monai.data.transforms.adaptor` to wrap transforms that don't conform
    to the requirements. This approach allows you to use transforms from
    otherwise incompatible libraries with minimal additional work.

    Note:
    In many cases, Compose is not the best way to create pre-processing
    pipelines. Pre-processing is often not a strictly sequential series of
    operations, and much of the complexity arises when a not-sequential
    set of functions must be called as if it were a sequence.

    Example: images and labels
    Images typically require some kind of normalisation that labels do not.
    Both are then typically augmented through the use of random rotations,
    flips, and deformations.
    Compose can be used with a series of transforms that take a dictionary
    that contains 'image' and 'label' entries. This might require wrapping
    `torchvision` transforms before passing them to compose.
    Alternatively, one can create a class with a __call__ function that
    calls your pre-processing functions taking into account that not all of
    them are called on the labels

    TODO: example / links to alternative approaches

    """

    def __init__(self, transforms=None):
        if transforms is None:
            transforms = []
        if not isinstance(transforms, (list, tuple)):
            raise ValueError("Parameters 'transforms' must be a list or tuple")
        self.transforms = transforms

    def __call__(self, input_):
        for transform in self.transforms:
            input_ = transform(input_)
        return input_


class Randomizable:
    """
    An interface for handling local numpy random state.
    this is mainly for randomized data augmentation transforms.
    """
    R = np.random.RandomState()

    def set_random_state(self, seed=None, state=None):
        """
        Set the random state locally, to control the randomness, the derived
        classes should use `self.R` instead of `np.random` to introduce random
        factors.

        Args:
            seed (int): set the random state with an integer seed.
            state (np.random.RandomState): set the random state with a `np.random.RandomState` object.

        Note:
            thread safety
        """
        if seed is not None:
            _seed = id(seed) if not isinstance(seed, int) else seed
            self.R = np.random.RandomState(_seed)
            return

        if state is not None:
            if not isinstance(state, np.random.RandomState):
                raise ValueError('`state` must be a `np.random.RandomState`, got {}'.format(type(state)))
            self.R = state
            return

        self.R = np.random.RandomState()
        return

    def randomise(self):
        """
        all self.R calls happen here so that we have a better chance to identify errors of sync the random state.
        """
        raise NotImplementedError

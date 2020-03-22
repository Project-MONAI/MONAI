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
"""
A collection of generic interfaces for MONAI transforms.
"""

import warnings
from typing import Hashable

import numpy as np

from monai.utils.misc import ensure_tuple


class Transform:
    """
    An abstract class of a ``Transform``.
    A transform is callable that processes ``data``.

    It could be stateful and may modify ``data`` in place,
    the implementation should be aware of:

        #. thread safety when mutating its own states.
           When used from a multi-process context, transform's instance variables are read-only.
        #. ``data`` content unused by this transform may still be used in the
           subsequent transforms in a composed transform.
        #. storing too much information in ``data`` may not scale.

    See Also

        :py:class:`monai.transforms.compose.Compose`
    """

    def __call__(self, data):
        """
        ``data`` is an element which often comes from an iteration over an
        iterable, such as :py:class:`torch.utils.data.Dataset`. This method should
        return an updated version of ``data``.
        To simplify the input validations, most of the transforms assume that

        - ``data`` component is a "channel-first" array,
        - the channel dimension is not omitted even if number of channels is one.
        """
        raise NotImplementedError


class Randomizable:
    """
    An interface for handling local numpy random state.
    this is mainly for randomized data augmentation transforms.
    """
    R = np.random.RandomState()

    def set_random_state(self, seed=None, state=None):
        """
        Set the random state locally, to control the randomness, the derived
        classes should use :py:attr:`self.R` instead of `np.random` to introduce random
        factors.

        Args:
            seed (int): set the random state with an integer seed.
            state (np.random.RandomState): set the random state with a `np.random.RandomState` object.

        Returns:
            a Randomizable instance.
        """
        if seed is not None:
            _seed = id(seed) if not isinstance(seed, int) else seed
            self.R = np.random.RandomState(_seed)
            return self

        if state is not None:
            if not isinstance(state, np.random.RandomState):
                raise ValueError('`state` must be a `np.random.RandomState`, got {}'.format(type(state)))
            self.R = state
            return self

        self.R = np.random.RandomState()
        return self

    def randomize(self):
        """
        Within this method, :py:attr:`self.R` should be used, instead of `np.random`, to introduce random factors.

        all :py:attr:`self.R` calls happen here so that we have a better chance to
        identify errors of sync the random state.
        """
        raise NotImplementedError


class Compose(Randomizable):
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
        Images typically require some kind of normalisation that labels do not.
        Both are then typically augmented through the use of random rotations,
        flips, and deformations.
        Compose can be used with a series of transforms that take a dictionary
        that contains 'image' and 'label' entries. This might require wrapping
        `torchvision` transforms before passing them to compose.
        Alternatively, one can create a class with a `__call__` function that
        calls your pre-processing functions taking into account that not all of
        them are called on the labels.
    """

    def __init__(self, transforms=None):
        if transforms is None:
            transforms = []
        if not isinstance(transforms, (list, tuple)):
            raise ValueError("Parameters 'transforms' must be a list or tuple")
        self.transforms = transforms

    def set_random_state(self, seed=None, state=None):
        for _transform in self.transforms:
            if not isinstance(_transform, Randomizable):
                continue
            _transform.set_random_state(seed, state)

    def randomize(self):
        for _transform in self.transforms:
            if not isinstance(_transform, Randomizable):
                continue
            try:
                _transform.randomize()
            except TypeError as type_error:
                warnings.warn(
                    'Transform "{0}" in Compose not randomized\n{0}.{1}.'.format(type(_transform).__name__, type_error),
                    RuntimeWarning)

    def __call__(self, input_):
        for transform in self.transforms:
            # if some transform generated batch list of data in the transform chain,
            # all the following transforms should apply to every item of the list.
            if isinstance(input_, list):
                for i, item in enumerate(input_):
                    input_[i] = transform(item)
            else:
                input_ = transform(input_)
        return input_


class MapTransform(Transform):
    """
    A subclass of :py:class:`monai.transforms.compose.Transform` with an assumption
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
                        # do nothing or some exceptions handling.
                return data

    """

    def __init__(self, keys):
        self.keys = ensure_tuple(keys)
        if not self.keys:
            raise ValueError('keys unspecified')
        for key in self.keys:
            if not isinstance(key, Hashable):
                raise ValueError('keys should be a hashable or a sequence of hashables, got {}'.format(type(key)))

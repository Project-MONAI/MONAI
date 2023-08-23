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

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy._typing import _ArrayLikeFloat_co, _ArrayLikeInt_co

from monai.config.type_definitions import DtypeLike, NdarrayTensor, ShapeLike


# FIXME What should the type be for the array in and out?
@runtime_checkable
class SupportsRandomGeneration(Protocol):
    def integers(
        self, low: int, high: int | None = None, size: ShapeLike | None = None, dtype=DtypeLike, endpoint: bool = False
    ) -> NdarrayTensor:
        ...

    def random(self, size: ShapeLike | None = None, dtype=DtypeLike, out: NdarrayTensor | None = None) -> NdarrayTensor:
        ...

    def choice(
        self,
        a: NdarrayTensor,
        size: ShapeLike | None = None,
        replace: bool = True,
        p: NdarrayTensor | None = None,
        axis: int = 0,
        shuffle: bool = True,
    ) -> NdarrayTensor:
        ...

    def bytes(self, length: int) -> str:
        ...

    def shuffle(self, x: NdarrayTensor, axis: int = 0) -> None:
        ...

    def permutation(self, x: NdarrayTensor, axis: int = 0) -> NdarrayTensor:
        ...

    def multinomial(
        self, n: _ArrayLikeInt_co, pvals: _ArrayLikeFloat_co, size: ShapeLike | None = None
    ) -> NdarrayTensor:
        ...

    def normal(self, loc: float = 0.0, scale: float = 1.0, size: ShapeLike | None = None) -> NdarrayTensor:
        ...

    def uniform(self, low: float = 0.0, high: float = 1.0, size: ShapeLike | None = None) -> NdarrayTensor:
        ...


class LegacyRandomStateAdaptor(SupportsRandomGeneration):
    random_staters: np.random.RandomState

    def __init__(self, seed: int | None = None, random_state: np.random.RandomState | None = None):
        if random_state is not None and seed is not None:
            raise ValueError("Cannot specify both rs and seed.")
        self.random_state = np.random.RandomState(seed=seed) if random_state is None else random_state

    def integers(
        self, low: int, high: int | None = None, size: ShapeLike | None = None, dtype=DtypeLike, endpoint: bool = False
    ) -> NdarrayTensor:
        return self.random_state.randint(low=low, high=high if endpoint else high + 1, size=size, dtype=dtype)

    def random(self, size: ShapeLike | None = None, dtype=DtypeLike, out: NdarrayTensor | None = None) -> NdarrayTensor:
        if out is not None:
            raise NotImplementedError("out is not implemented")
        if dtype is not None:
            raise NotImplementedError("dtype is not implemented")
        return self.random_state.random(size)

    def choice(
        self,
        a: NdarrayTensor,
        size: ShapeLike | None = None,
        replace: bool = True,
        p: NdarrayTensor | None = None,
        axis: int = 0,
        shuffle: bool = True,
    ) -> NdarrayTensor:
        if axis != 0:
            raise NotImplementedError("axis is not implemented")
        if not shuffle:
            raise NotImplementedError("shuffle is not implemented")
        return self.random_state.choice(a, size, replace, p)

    def permutation(self, x: NdarrayTensor, axis: int = 0) -> NdarrayTensor:
        if axis != 0:
            raise NotImplementedError("axis is not implemented")
        return self.random_state.permutation(x)

    def shuffle(self, x: NdarrayTensor, axis: int = 0) -> None:
        if axis != 0:
            raise NotImplementedError("axis is not implemented")
        return self.random_state.shuffle(x)

    def multinomial(
        self, n: _ArrayLikeInt_co, pvals: _ArrayLikeFloat_co, size: ShapeLike | None = None
    ) -> NdarrayTensor:
        return self.random_state.multinomial(n, pvals, size)

    def normal(self, loc: float = 0.0, scale: float = 1.0, size: ShapeLike | None = None) -> NdarrayTensor:
        return self.random_state.normal(loc, scale, size)

    def uniform(self, low: float = 0.0, high: float = 1.0, size: ShapeLike | None = None) -> NdarrayTensor:
        return self.random_state.uniform(low, high, size)


def handle_legacy_random_state(
    rand_state: np.random.RandomState | None = None,
    generator: SupportsRandomGeneration | None = None,
    return_legacy_default_random: bool = False,
) -> SupportsRandomGeneration | None:
    if generator is not None and rand_state is not None:
        raise ValueError("rand_state and generator cannot be set at the same time.")

    if rand_state is not None:
        generator = rand_state
        rand_state = None
    if isinstance(generator, np.random.RandomState):
        generator = LegacyRandomStateAdaptor(generator)

    if generator is None and return_legacy_default_random:
        generator = LegacyRandomStateAdaptor(np.random.random.__self__)

    return generator

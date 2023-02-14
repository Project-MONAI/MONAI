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

from typing import Any, Union, Tuple

import numpy as np


def validate_compatible_scalar_or_tuple(
        left_name,
        right_name,
        left_value,
        right_value
):
    if ((isinstance(left_value, tuple) and not isinstance(right_value, tuple)) or
            (not isinstance(left_value, tuple) and isinstance(right_value, tuple))):
        raise ValueError(f"if '{left_name}' is a tuple '{right_name}' must also "
                         "be a tuple")

    if isinstance(left_value, tuple):
        if len(left_value) != len(right_value):
            raise ValueError(f"'{left_name}' and '{right_name}' must be the same "
                             "length if they are tuples but are length "
                             f"{len(left_value)} and {len(right_value)} respectively")


class Randomizer:

    def __init__(
            self,
            prob: float = 1.0,
            seed=None,
            state=None
    ):
        self.R = None
        self.set_random_state(seed, state)

        if not 0.0 <= prob <= 1.0:
            raise ValueError(f"'prob' must be between 0.0 and 1.0 inclusive but is {prob}")
        self.prob = prob

    def set_random_state(self, seed=None, state=None):
        if seed is not None:
            self.R = np.random.RandomState(seed)
        elif state is not None:
            self.R = state
        else:
            self.R = np.random.RandomState()

    def do_random(self):
        return self.R.uniform() <= self.prob

    def sample(self, *args, **kwargs):
        return self.R.uniform()


class BooleanRandomizer(Randomizer):
    def __init__(
            self,
            threshold: Union[Tuple[float], float] = 0.5,
            prob: float = 1.0,
            default: Union[Tuple[Any], Any] = False,
            seed: int | None = None,
            state: np.random.RandomState = None
    ):
        super().__init__(prob, state, seed)

        validate_compatible_scalar_or_tuple('threshold', 'default', threshold, default)

        self.threshold = threshold
        self.default = default

    def sample(self):
        if self.do_random():
            if isinstance(self.threshold, tuple):
                return tuple(self.R.uniform() <= t for t in self.threshold)
            else:
                return self.R.uniform() <= self.threshold


class DiscreteRandomizer(Randomizer):
    def __init__(
            self,
            min_value,
            max_value,
            prob: float = 1.0,
            default: Any = 0,
            seed: int | None = None,
            state: np.random.RandomState | None = None
    ):
        super().__init__(prob, state, seed)

        validate_compatible_scalar_or_tuple('min_value', 'max_value', min_value, max_value)
        validate_compatible_scalar_or_tuple('min_value', 'default', min_value, default)

        self.min_value = min_value
        self.max_value = max_value
        self.default = default

    def sample(
            self
    ):
        if self.do_random():
            if isinstance(self.min_value, tuple):
                return tuple(self.R.randint(v_min, v_max + 1)
                             for v_min, v_max in zip(self.min_value, self.max_value))
            else:
                return self.R.randint(self.min_value, self.max_value + 1)

        return self.default


class ContinuousRandomizer(Randomizer):
    def __init__(
            self,
            min_value,
            max_value,
            prob,
            default: Any = 0.0,
            seed: int | None = None,
            state: np.random.RandomState | None = None
    ):
        super().__init__(prob, seed, state)

        validate_compatible_scalar_or_tuple('min_value', 'max_value', min_value, max_value)
        if isinstance(min_value, tuple):
            default_ = tuple(default for _ in min_value)
        else:
            # validate_compatible_scalar_or_tuple('min_value', 'default', min_value, default)
            default_ = default

        self.min_value = min_value
        self.max_value = max_value
        self.default = default_

    def sample(self):
        if self.do_random():
            if isinstance(self.min_value, tuple):
                return tuple(self.R.uniform(v_min, v_max)
                             for v_min, v_max in zip(self.min_value, self.max_value))
            else:
                return self.R.uniform(self.min_value, self.max_value)

        return self.default

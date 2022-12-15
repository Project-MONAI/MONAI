from typing import Any, Tuple, Union

import numpy as np

import torch
from nptyping import Integer


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
        return self.R.uniform() < self.prob

    def sample(self, *args, **kwargs):
        return self.R.uniform()


class RotateRandomizer(Randomizer):

    def __init__(
            self,
            range_x,
            range_y,
            range_z,
            prob: float = 1.0,
            seed=None,
            state=None,
    ):
        super().__init__(prob, state, seed)
        self.range_x = range_x
        self.range_y = range_y
        self.range_z = range_z

    def sample(
            self,
            data: torch.Tensor = None
    ):
        if not isinstance(data, (np.ndarray, torch.Tensor)):
            raise ValueError("data must be a numpy ndarray or torch tensor but is of "
                             f"type {type(data)}")

        spatial_shape = len(data.shape[1:])
        if spatial_shape == 2:
            if self.do_random():
                return self.R.uniform(self.range_x[0], self.range_x[1])
            return 0.0
        elif spatial_shape == 3:
            if self.do_random():
                x = self.R.uniform(self.range_x[0], self.range_x[1])
                y = self.R.uniform(self.range_y[0], self.range_y[1])
                z = self.R.uniform(self.range_z[0], self.range_z[1])
                return x, y, z
            return 0.0, 0.0, 0.0
        else:
            raise ValueError("data should be a tensor with 2 or 3 spatial dimensions but it "
                             f"has {spatial_shape} spatial dimensions")


class BooleanRandomizer(Randomizer):
    def __init__(
            self,
            threshold: Union[Tuple[float], float] = 1.0,
            prob: float = 1.0,
            default: Union[Tuple[Any], Any] = False,
            seed: Integer = None,
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
            seed: Integer = None,
            state: np.random.RandomState = None
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


class SpatialAxisRandomizer(Randomizer):
    def __init__(
            self,
            prob: float = 1.0,
            default: Any = 0,
            seed: Integer = None,
            state: np.random.RandomState = None
    ):
        super().__init__(prob, state, seed)
        self.default = default

    def sample(
            self,
            data: torch.Tensor
    ):
        data_spatial_dims = len(data.shape) - 1
        if self.do_random():
            if isinstance(self.default, tuple):
                return tuple(self.R.randint(0, data_spatial_dims + 1)
                             for _ in self.default)
            else:
                return self.R.randint(0, data_spatial_dims + 1)

        return self.default


class ContinuousRandomizer(Randomizer):
    def __init__(
            self,
            min_value,
            max_value,
            prob,
            default: Any = 0.0,
            seed: Integer = None,
            state: np.random.RandomState = None
    ):
        super().__init__(prob, state, seed)

        validate_compatible_scalar_or_tuple('min_value', 'max_value', min_value, max_value)
        validate_compatible_scalar_or_tuple('min_value', 'default', min_value, default)

        self.min_value = min_value
        self.max_value = max_value
        self.default = default

    def sample(self):
        if self.do_random():
            if isinstance(self.min_value, tuple):
                return tuple(self.R.uniform(v_min, v_max)
                             for v_min, v_max in zip(self.min_value, self.max_value))
            else:
                return self.R.uniform(self.min_value, self.max_value)

        return self.default


class Elastic3DRandomizer(Randomizer):

    def __init__(
        self,
        sigma_range,
        magnitude_range,
        prob=1.0,
        grid_size=None,
        seed: Integer=None,
        state: np.random.RandomState=None
    ):
        super().__init__(prob, seed, state)
        self.grid_size = grid_size
        self.sigma_range = sigma_range
        self.magnitude_range = magnitude_range

    def sample(
            self,
            grid_size,
            device
    ):
        if self.do_random():
            rand_offsets = self.R.uniform(-1.0, 1.0, [3] + list(grid_size)).astype(np.float32, copy=False)
            rand_offsets = torch.as_tensor(rand_offsets, device=device).unsqueeze(0)
            sigma = self.R.uniform(self.sigma_range[0], self.sigma_range[1])
            magnitude = self.R.uniform(self.magnitude_range[0], self.magnitude_range[1])
            return rand_offsets, magnitude, sigma

        return None, None, None

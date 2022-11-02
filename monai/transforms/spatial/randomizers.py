import numpy as np

import torch


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

    def sample(self):
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


class Elastic3DRandomizer(Randomizer):

    def __init__(
        self,
        sigma_range,
        magnitude_range,
        prob=1.0,
        grid_size=None,
        seed=None,
        state=None,
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

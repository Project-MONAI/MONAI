from __future__ import annotations

from typing import Sequence

import numpy as np

from monai.transforms.utility.randomizer import Randomizer


class CropRandomizer(Randomizer):

    def __init__(
            self,
            sizes: Sequence[int] | int,
            prob: float = 1.0,
            seed: int | None = None,
            state: np.random.RandomState | None = None
    ):
        super().__init__(prob, seed, state)

        self.sizes = sizes

    def do_random(self):
        return True

    def sample(self, shape: Sequence[int]):
        if self.do_random():
            if isinstance(self.sizes, int):
                crop_shape = tuple(self.sizes for _ in shape)
            else:
                crop_shape = self.sizes

            if len(shape) != len(crop_shape):
                raise ValueError("self.sizes and shape must be the same length if self.sizes "
                                 "is initialized with a tuple "
                                 f" (lengths {len(self.sizes)} and {len(shape)} respectively)")

            valid_ranges = tuple(i - c for i, c in zip(shape, crop_shape))

            values = tuple(self.R.randint(0, r + 1) for r in valid_ranges)
            starts = tuple(v if r > 0 else 0 for v, r in zip(values, valid_ranges))
            # starts = tuple(self.R.randint(0, r + 1 if r > 0 else 0) for r in valid_ranges)
            slices = tuple(slice(s, s + c) for s, c in zip(starts, crop_shape))
            return slices

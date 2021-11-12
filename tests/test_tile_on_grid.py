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

import unittest
from typing import Optional

import numpy as np
from parameterized import parameterized

from monai.apps.pathology.transforms import TileOnGrid

TEST_CASES = []
for tile_count in [16, 64]:
    for tile_size in [8, 32]:
        for filter_mode in ["min", "max", "random"]:
            for background_val in [255, 0]:
                TEST_CASES.append(
                    [
                        {
                            "tile_count": tile_count,
                            "tile_size": tile_size,
                            "filter_mode": filter_mode,
                            "random_offset": False,
                            "background_val": background_val,
                        }
                    ]
                )

TEST_CASES2 = []
for tile_count in [16, 64]:
    for tile_size in [8, 32]:
        for filter_mode in ["min", "max", "random"]:
            for background_val in [255, 0]:
                TEST_CASES2.append(
                    [
                        {
                            "tile_count": tile_count,
                            "tile_size": tile_size,
                            "filter_mode": filter_mode,
                            "random_offset": True,
                            "background_val": background_val,
                        }
                    ]
                )


def make_image(
    tile_count: int, tile_size: int, random_offset: bool = False, filter_mode: Optional[str] = None, seed=123, **kwargs
):

    tile_count = int(np.sqrt(tile_count))
    pad = 0
    if random_offset:
        pad = 3

    image = np.random.randint(200, size=[3, tile_count * tile_size + pad, tile_count * tile_size + pad], dtype=np.uint8)
    imlarge = image

    random_state = np.random.RandomState(seed)

    if random_offset:
        image = image[
            :, random_state.randint(image.shape[1] % tile_size) :, random_state.randint(image.shape[2] % tile_size) :
        ]

    tiles_list = []
    for x in range(tile_count):
        for y in range(tile_count):
            tiles_list.append(image[:, x * tile_size : (x + 1) * tile_size, y * tile_size : (y + 1) * tile_size])

    tiles = np.stack(tiles_list, axis=0)  # type: ignore

    if filter_mode == "min" or filter_mode == "max":
        tiles = tiles[np.argsort(tiles.sum(axis=(1, 2, 3)))]

    return imlarge, tiles


class TestTileOnGrid(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_tile_pathce_single_call(self, input_parameters):

        img, tiles = make_image(**input_parameters)

        tiler = TileOnGrid(**input_parameters)
        output = tiler(img)
        np.testing.assert_equal(output, tiles)

    @parameterized.expand(TEST_CASES2)
    def test_tile_pathce_random_call(self, input_parameters):

        img, tiles = make_image(**input_parameters, seed=123)

        tiler = TileOnGrid(**input_parameters)
        tiler.set_random_state(seed=123)

        output = tiler(img)
        np.testing.assert_equal(output, tiles)


if __name__ == "__main__":
    unittest.main()

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

import unittest
from typing import Optional

import numpy as np
import torch
from parameterized import parameterized

from monai.apps.pathology.transforms import TileOnGridDict
from tests.utils import TEST_NDARRAYS, assert_allclose

TEST_CASES = []
for tile_count in [16, 64]:
    for tile_size in [8, 32]:
        for filter_mode in ["min", "max", "random"]:
            for background_val in [255, 0]:
                for return_list_of_dicts in [False, True]:
                    TEST_CASES.append(
                        [
                            {
                                "tile_count": tile_count,
                                "tile_size": tile_size,
                                "filter_mode": filter_mode,
                                "random_offset": False,
                                "background_val": background_val,
                                "return_list_of_dicts": return_list_of_dicts,
                            }
                        ]
                    )

for tile_size in [8, 16]:
    for step in [4, 8]:
        TEST_CASES.append([{"tile_count": 16, "step": step, "tile_size": tile_size}])

TESTS = []
for p in TEST_NDARRAYS:
    for tc in TEST_CASES:
        TESTS.append([p, *tc])

TEST_CASES2 = []
for tile_count in [16, 64]:
    for tile_size in [8, 32]:
        for filter_mode in ["min", "max", "random"]:
            for background_val in [255, 0]:
                for return_list_of_dicts in [False, True]:
                    TEST_CASES2.append(
                        [
                            {
                                "tile_count": tile_count,
                                "tile_size": tile_size,
                                "filter_mode": filter_mode,
                                "random_offset": True,
                                "background_val": background_val,
                                "return_list_of_dicts": return_list_of_dicts,
                            }
                        ]
                    )

TESTS2 = []
for p in TEST_NDARRAYS:
    for tc in TEST_CASES2:
        TESTS2.append([p, *tc])

for tile_size in [8, 16]:
    for step in [4, 8]:
        TEST_CASES.append([{"tile_count": 16, "step": step, "tile_size": tile_size}])


def make_image(
    tile_count: int,
    tile_size: int,
    step: int = 0,
    random_offset: bool = False,
    filter_mode: Optional[str] = None,
    seed=123,
    **kwargs,
):

    tile_count = int(np.sqrt(tile_count))
    pad = 0
    if random_offset:
        pad = 3

    if step == 0:
        step = tile_size

    image = np.random.randint(
        200,
        size=[3, (tile_count - 1) * step + tile_size + pad, (tile_count - 1) * step + tile_size + pad],
        dtype=np.uint8,
    )
    imlarge = image

    random_state = np.random.RandomState(seed)

    if random_offset:
        pad_h = image.shape[1] % tile_size
        pad_w = image.shape[2] % tile_size
        offset = (random_state.randint(pad_h) if pad_h > 0 else 0, random_state.randint(pad_w) if pad_w > 0 else 0)
        image = image[:, offset[0] :, offset[1] :]

    tiles_list = []
    for x in range(tile_count):
        for y in range(tile_count):
            tiles_list.append(image[:, x * step : x * step + tile_size, y * step : y * step + tile_size])

    tiles = np.stack(tiles_list, axis=0)

    if (filter_mode == "min" or filter_mode == "max") and len(tiles) > tile_count**2:
        tiles = tiles[np.argsort(tiles.sum(axis=(1, 2, 3)))]

    return imlarge, tiles


class TestTileOnGridDict(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_tile_patch_single_call(self, in_type, input_parameters):

        key = "image"
        input_parameters["keys"] = key

        img, tiles = make_image(**input_parameters)
        input_img = in_type(img)

        splitter = TileOnGridDict(**input_parameters)

        output = splitter({key: input_img})

        if input_parameters.get("return_list_of_dicts", False):
            if isinstance(input_img, torch.Tensor):
                output = torch.stack([ix[key] for ix in output], axis=0)
            else:
                output = np.stack([ix[key] for ix in output], axis=0)
        else:
            output = output[key]

        assert_allclose(output, tiles, type_test=False)

    @parameterized.expand(TESTS2)
    def test_tile_patch_random_call(self, in_type, input_parameters):

        key = "image"
        input_parameters["keys"] = key

        random_state = np.random.RandomState(123)
        seed = random_state.randint(10000)
        img, tiles = make_image(**input_parameters, seed=seed)
        input_img = in_type(img)

        splitter = TileOnGridDict(**input_parameters)
        splitter.set_random_state(seed=123)

        output = splitter({key: input_img})

        if input_parameters.get("return_list_of_dicts", False):
            if isinstance(input_img, torch.Tensor):
                output = torch.stack([ix[key] for ix in output], axis=0)
            else:
                output = np.stack([ix[key] for ix in output], axis=0)
        else:
            output = output[key]
        assert_allclose(output, tiles, type_test=False)


if __name__ == "__main__":
    unittest.main()

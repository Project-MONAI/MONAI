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

import os
import tempfile
import unittest

import numpy as np
from parameterized import parameterized
from PIL import Image

from monai.data import write_png
from tests.utils import TEST_NDARRAYS

TESTS = []
for p in TEST_NDARRAYS:
    TESTS.append((p,))
# TESTS = [[np.array]]


class TestPngWrite(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_write_gray(self, in_type):
        with tempfile.TemporaryDirectory() as out_dir:
            image_name = os.path.join(out_dir, "test.png")
            img = np.random.rand(2, 3)
            img_save_val = (255 * img).astype(np.uint8)
            write_png(in_type(img), image_name, scale=255)
            out = np.asarray(Image.open(image_name))
            out = np.moveaxis(out, 0, 1)
            np.testing.assert_allclose(out, img_save_val)

    @parameterized.expand(TESTS)
    def test_write_gray_1height(self, in_type):
        with tempfile.TemporaryDirectory() as out_dir:
            image_name = os.path.join(out_dir, "test.png")
            img = np.random.rand(1, 3)
            img_save_val = (65535 * img).astype(np.uint16)
            write_png(in_type(img), image_name, scale=65535)
            out = np.asarray(Image.open(image_name))
            out = np.moveaxis(out, 0, 1)
            np.testing.assert_allclose(out, img_save_val)

    @parameterized.expand(TESTS)
    def test_write_gray_1channel(self, in_type):
        with tempfile.TemporaryDirectory() as out_dir:
            image_name = os.path.join(out_dir, "test.png")
            img = np.random.rand(2, 3, 1)
            img_save_val = (255 * img).astype(np.uint8).squeeze(2)
            write_png(in_type(img), image_name, scale=255)
            out = np.asarray(Image.open(image_name))
            out = np.moveaxis(out, 0, 1)
            np.testing.assert_allclose(out, img_save_val)

    @parameterized.expand(TESTS)
    def test_write_rgb(self, in_type):
        with tempfile.TemporaryDirectory() as out_dir:
            image_name = os.path.join(out_dir, "test.png")
            img = np.random.rand(2, 3, 3)
            img_save_val = (255 * img).astype(np.uint8)
            write_png(in_type(img), image_name, scale=255)
            out = np.asarray(Image.open(image_name))
            out = np.moveaxis(out, 0, 1)
            np.testing.assert_allclose(out, img_save_val)

    @parameterized.expand(TESTS)
    def test_write_2channels(self, in_type):
        with tempfile.TemporaryDirectory() as out_dir:
            image_name = os.path.join(out_dir, "test.png")
            img = np.random.rand(2, 3, 2)
            img_save_val = (255 * img).astype(np.uint8)
            write_png(in_type(img), image_name, scale=255)
            out = np.asarray(Image.open(image_name))
            out = np.moveaxis(out, 0, 1)
            np.testing.assert_allclose(out, img_save_val)

    @parameterized.expand(TESTS)
    def test_write_output_shape(self, in_type):
        with tempfile.TemporaryDirectory() as out_dir:
            image_name = os.path.join(out_dir, "test.png")
            img = np.random.rand(2, 2, 3)
            write_png(in_type(img), image_name, (4, 4), scale=255)
            out = np.asarray(Image.open(image_name))
            np.testing.assert_allclose(out.shape, (4, 4, 3))


if __name__ == "__main__":
    unittest.main()

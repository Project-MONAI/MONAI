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

import os
import tempfile
import unittest
from unittest.case import skipUnless

import numpy as np
from parameterized import parameterized

from monai.data import NrrdReader
from monai.utils.module import optional_import

nrrd, has_nrrd = optional_import("nrrd", allow_namespace_pkg=True)

TEST_CASE_1 = [(4, 4), "test_image.nrrd", (4, 4), np.uint8]
TEST_CASE_2 = [(4, 4, 4), "test_image.nrrd", (4, 4, 4), np.uint16]
TEST_CASE_3 = [(4, 4, 4, 4), "test_image.nrrd", (4, 4, 4, 4), np.uint32]
TEST_CASE_4 = [(1, 2, 3, 4, 5), "test_image.nrrd", (1, 2, 3, 4, 5), np.uint64]
TEST_CASE_5 = [(6, 5, 4, 3, 2, 1), "test_image.nrrd", (6, 5, 4, 3, 2, 1), np.float32]
TEST_CASE_6 = [(4,), "test_image.nrrd", (4,), np.float64]
TEST_CASE_7 = [(4, 4), ["test_image.nrrd", "test_image2.nrrd", "test_image3.nrrd"], (4, 4), np.float32]
TEST_CASE_8 = [
    (3, 4, 4, 1),
    "test_image.nrrd",
    (3, 4, 4, 1),
    np.float32,
    {
        "dimension": 4,
        "space": "left-posterior-superior",
        "sizes": [3, 4, 4, 1],
        "space directions": [[0.7, 0.0, 0.0], [0.0, 0.0, -0.8], [0.0, 0.9, 0.0]],
        "space origin": [1.0, 5.0, 20.0],
    },
]
# 4-D NRRD with an explicit 'list' channel axis (kinds: list domain domain domain).
# pynrrd stores the 'none' space direction for the channel axis as a row of NaN values.
TEST_CASE_4D_CHANNEL = [
    (3, 4, 5, 6),  # (channel, H, W, D)
    "test_4d_channel.nrrd",
    np.float32,
    {
        "dimension": 4,
        "space": "left-posterior-superior",
        "kinds": ["list", "domain", "domain", "domain"],
        "sizes": [3, 4, 5, 6],
        "space directions": np.array([[np.nan, np.nan, np.nan], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]),
        "space origin": np.array([10.0, 20.0, 30.0]),
    },
]


@skipUnless(has_nrrd, "nrrd required")
class TestNrrdReader(unittest.TestCase):

    def test_verify_suffix(self):
        reader = NrrdReader()
        self.assertFalse(reader.verify_suffix("test_image.nrd"))
        reader.verify_suffix("test_image.nrrd")
        reader.verify_suffix("test_image.seg.nrrd")

    @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_3, TEST_CASE_4])
    def test_read_int(self, data_shape, filename, expected_shape, dtype):
        min_val, max_val = np.iinfo(dtype).min, np.iinfo(dtype).max
        test_image = np.random.randint(min_val, max_val, size=data_shape, dtype=dtype)
        with tempfile.TemporaryDirectory() as tempdir:
            filename = os.path.join(tempdir, filename)
            nrrd.write(filename, test_image.astype(dtype))
            reader = NrrdReader()
            result = reader.read(filename)
        self.assertEqual(result.array.dtype, dtype)
        self.assertTupleEqual(result.array.shape, expected_shape)
        self.assertTupleEqual(tuple(result.header["sizes"]), expected_shape)
        np.testing.assert_allclose(result.array, test_image)

    @parameterized.expand([TEST_CASE_5, TEST_CASE_6])
    def test_read_float(self, data_shape, filename, expected_shape, dtype):
        test_image = np.random.rand(*data_shape).astype(dtype)
        with tempfile.TemporaryDirectory() as tempdir:
            filename = os.path.join(tempdir, filename)
            nrrd.write(filename, test_image.astype(dtype))
            reader = NrrdReader()
            result = reader.read(filename)
        self.assertEqual(result.array.dtype, dtype)
        self.assertTupleEqual(result.array.shape, expected_shape)
        self.assertTupleEqual(tuple(result.header["sizes"]), expected_shape)
        np.testing.assert_allclose(result.array, test_image)

    @parameterized.expand([TEST_CASE_7])
    def test_read_list(self, data_shape, filenames, expected_shape, dtype):
        test_image = np.random.rand(*data_shape).astype(dtype)
        with tempfile.TemporaryDirectory() as tempdir:
            for i, filename in enumerate(filenames):
                filenames[i] = os.path.join(tempdir, filename)
                nrrd.write(filenames[i], test_image.astype(dtype))
            reader = NrrdReader()
            results = reader.read(filenames)
        for result in results:
            self.assertTupleEqual(result.array.shape, expected_shape)
            self.assertTupleEqual(tuple(result.header["sizes"]), expected_shape)
            np.testing.assert_allclose(result.array, test_image)

    @parameterized.expand([TEST_CASE_8])
    def test_read_with_header(self, data_shape, filename, expected_shape, dtype, reference_header):
        test_image = np.random.rand(*data_shape).astype(dtype)
        with tempfile.TemporaryDirectory() as tempdir:
            filename = os.path.join(tempdir, filename)
            nrrd.write(filename, test_image.astype(dtype), header=reference_header)
            reader = NrrdReader()
            image_array, image_header = reader.get_data(reader.read(filename))
        self.assertIsInstance(image_array, np.ndarray)
        self.assertEqual(image_array.dtype, dtype)
        self.assertTupleEqual(image_array.shape, expected_shape)
        np.testing.assert_allclose(image_array, test_image)
        self.assertIsInstance(image_header, dict)
        self.assertTupleEqual(tuple(image_header["spatial_shape"]), expected_shape)
        np.testing.assert_allclose(
            image_header["affine"],
            np.array([[-0.7, 0.0, 0.0, -1.0], [0.0, 0.0, -0.9, -5.0], [0.0, -0.8, 0.0, 20.0], [0.0, 0.0, 0.0, 1.0]]),
        )

    @parameterized.expand([TEST_CASE_8])
    def test_read_with_header_index_order_c(self, data_shape, filename, expected_shape, dtype, reference_header):
        test_image = np.random.rand(*data_shape).astype(dtype)
        with tempfile.TemporaryDirectory() as tempdir:
            filename = os.path.join(tempdir, filename)
            nrrd.write(filename, test_image.astype(dtype), header=reference_header)
            reader = NrrdReader(index_order="C")
            image_array, image_header = reader.get_data(reader.read(filename))
        self.assertIsInstance(image_array, np.ndarray)
        self.assertEqual(image_array.dtype, dtype)
        self.assertTupleEqual(image_array.shape, expected_shape[::-1])
        self.assertTupleEqual(image_array.shape, tuple(image_header["spatial_shape"]))

    @parameterized.expand([TEST_CASE_4D_CHANNEL])
    def test_read_4d_channel(self, data_shape, filename, dtype, reference_header):
        """4-D NRRD with a 'list' channel axis must not crash in _get_affine and must
        set ORIGINAL_CHANNEL_DIM / spatial_shape correctly."""
        test_image = np.random.rand(*data_shape).astype(dtype)
        with tempfile.TemporaryDirectory() as tempdir:
            filepath = os.path.join(tempdir, filename)
            nrrd.write(filepath, test_image, header=reference_header)
            reader = NrrdReader()
            image_array, image_header = reader.get_data(reader.read(filepath))
        self.assertIsInstance(image_array, np.ndarray)
        self.assertEqual(image_array.dtype, dtype)
        self.assertTupleEqual(image_array.shape, data_shape)
        # spatial_shape must exclude the channel axis
        self.assertTupleEqual(tuple(image_header["spatial_shape"]), data_shape[1:])
        # channel dim 0 must be identified
        self.assertEqual(image_header["original_channel_dim"], 0)
        # affine must be a valid 4×4 matrix (3 spatial dims → 4×4)
        self.assertTupleEqual(image_header["affine"].shape, (4, 4))
        np.testing.assert_allclose(
            image_header["affine"],
            np.array([[-1.0, 0.0, 0.0, -10.0], [0.0, -2.0, 0.0, -20.0], [0.0, 0.0, 3.0, 30.0], [0.0, 0.0, 0.0, 1.0]]),
        )


if __name__ == "__main__":
    unittest.main()

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

from monai.transforms import LoadNumpy


class TestLoadNumpy(unittest.TestCase):
    def test_npy(self):
        test_data = np.random.randint(0, 256, size=[3, 4, 4])
        with tempfile.TemporaryDirectory() as tempdir:
            filepath = os.path.join(tempdir, "test_data.npy")
            np.save(filepath, test_data)

            result = LoadNumpy()(filepath)
        self.assertTupleEqual(result[1]["spatial_shape"], test_data.shape)
        self.assertTupleEqual(result[0].shape, test_data.shape)
        np.testing.assert_allclose(result[0], test_data)

    def test_npz1(self):
        test_data1 = np.random.randint(0, 256, size=[3, 4, 4])
        with tempfile.TemporaryDirectory() as tempdir:
            filepath = os.path.join(tempdir, "test_data.npy")
            np.save(filepath, test_data1)

            result = LoadNumpy()(filepath)
        self.assertTupleEqual(result[1]["spatial_shape"], test_data1.shape)
        self.assertTupleEqual(result[0].shape, test_data1.shape)
        np.testing.assert_allclose(result[0], test_data1)

    def test_npz2(self):
        test_data1 = np.random.randint(0, 256, size=[3, 4, 4])
        test_data2 = np.random.randint(0, 256, size=[3, 4, 4])
        with tempfile.TemporaryDirectory() as tempdir:
            filepath = os.path.join(tempdir, "test_data.npz")
            np.savez(filepath, test_data1, test_data2)

            result = LoadNumpy()(filepath)
        self.assertTupleEqual(result[1]["spatial_shape"], test_data1.shape)
        self.assertTupleEqual(result[0].shape, (2, 3, 4, 4))
        np.testing.assert_allclose(result[0], np.stack([test_data1, test_data2]))

    def test_npz3(self):
        test_data1 = np.random.randint(0, 256, size=[3, 4, 4])
        test_data2 = np.random.randint(0, 256, size=[3, 4, 4])
        with tempfile.TemporaryDirectory() as tempdir:
            filepath = os.path.join(tempdir, "test_data.npz")
            np.savez(filepath, test1=test_data1, test2=test_data2)

            result = LoadNumpy(npz_keys=["test1", "test2"])(filepath)
        self.assertTupleEqual(result[1]["spatial_shape"], test_data1.shape)
        self.assertTupleEqual(result[0].shape, (2, 3, 4, 4))
        np.testing.assert_allclose(result[0], np.stack([test_data1, test_data2]))

    def test_npy_pickle(self):
        test_data = {"test": np.random.randint(0, 256, size=[3, 4, 4])}
        with tempfile.TemporaryDirectory() as tempdir:
            filepath = os.path.join(tempdir, "test_data.npy")
            np.save(filepath, test_data, allow_pickle=True)

            result = LoadNumpy(data_only=True, dtype=None)(filepath).item()
        self.assertTupleEqual(result["test"].shape, test_data["test"].shape)
        np.testing.assert_allclose(result["test"], test_data["test"])


if __name__ == "__main__":
    unittest.main()

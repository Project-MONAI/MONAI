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

from monai.data import ITKReader, NibabelReader, NrrdReader, NumpyReader, PILReader, PydicomReader
from monai.transforms import LoadImage, LoadImaged
from tests.utils import SkipIfNoModule


class TestInitLoadImage(unittest.TestCase):
    def test_load_image(self):
        instance1 = LoadImage(image_only=False, dtype=None)
        instance2 = LoadImage(image_only=True, dtype=None)
        self.assertIsInstance(instance1, LoadImage)
        self.assertIsInstance(instance2, LoadImage)

        for r in ["NibabelReader", "PILReader", "ITKReader", "NumpyReader", "NrrdReader", "PydicomReader", None]:
            inst = LoadImaged("image", reader=r)
            self.assertIsInstance(inst, LoadImaged)

    @SkipIfNoModule("itk")
    @SkipIfNoModule("nibabel")
    @SkipIfNoModule("PIL")
    @SkipIfNoModule("nrrd")
    @SkipIfNoModule("Pydicom")
    def test_readers(self):
        inst = ITKReader()
        self.assertIsInstance(inst, ITKReader)

        inst = NibabelReader()
        self.assertIsInstance(inst, NibabelReader)
        inst = NibabelReader(as_closest_canonical=True)
        self.assertIsInstance(inst, NibabelReader)

        inst = PydicomReader()
        self.assertIsInstance(inst, PydicomReader)

        inst = NumpyReader()
        self.assertIsInstance(inst, NumpyReader)
        inst = NumpyReader(npz_keys="test")
        self.assertIsInstance(inst, NumpyReader)

        inst = PILReader()
        self.assertIsInstance(inst, PILReader)

        inst = NrrdReader()
        self.assertIsInstance(inst, NrrdReader)


if __name__ == "__main__":
    unittest.main()

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
import unittest
from unittest.mock import patch

from monai.data.image_reader import DICOM_READER_ENV_MAP, get_preferred_dicom_reader_key, is_dicom_path
from monai.transforms import LoadImage
from monai.transforms.io.array import SUPPORTED_READERS, get_default_reader_registration_order
from tests.test_utils import SkipIfNoModule


class TestNvImgCodecPydicomPlugin(unittest.TestCase):
    @SkipIfNoModule("pydicom")
    def test_is_dicom_path(self):
        self.assertTrue(is_dicom_path("tests/testing_data/CT_DICOM"))
        self.assertFalse(is_dicom_path("tests/testing_data/test_image.nii.gz"))

    def test_get_preferred_dicom_reader_key_default(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("MONAI_DICOM_READER", None)
            self.assertEqual(get_preferred_dicom_reader_key(), "")

    def test_get_preferred_dicom_reader_key_env(self):
        with patch.dict(os.environ, {"MONAI_DICOM_READER": "nvimgcodec"}):
            self.assertEqual(get_preferred_dicom_reader_key(), "nvimgcodecpydicomreader")
        with patch.dict(os.environ, {"MONAI_DICOM_READER": "pydicom"}):
            self.assertEqual(get_preferred_dicom_reader_key(), "pydicomreader")
        with patch.dict(os.environ, {"MONAI_DICOM_READER": "itk"}):
            self.assertEqual(get_preferred_dicom_reader_key(), "itkreader")

    def test_get_preferred_dicom_reader_key_invalid(self):
        with patch.dict(os.environ, {"MONAI_DICOM_READER": "unknown"}):
            self.assertEqual(get_preferred_dicom_reader_key(), "")

    def test_get_default_reader_registration_order_default(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("MONAI_DICOM_READER", None)
            order = get_default_reader_registration_order()
            self.assertEqual(
                order,
                [
                    "pydicomreader",
                    "nvimgcodecpydicomreader",
                    "itkreader",
                    "nrrdreader",
                    "numpyreader",
                    "pilreader",
                    "nibabelreader",
                ],
            )

    def test_get_default_reader_registration_order(self):
        with patch.dict(os.environ, {"MONAI_DICOM_READER": "pydicom"}):
            order = get_default_reader_registration_order()
            self.assertEqual(order[-1], "pydicomreader")
            self.assertIn("itkreader", order)
            self.assertIn("nvimgcodecpydicomreader", order)
            self.assertEqual(
                order[:-1],
                ["nvimgcodecpydicomreader", "itkreader", "nrrdreader", "numpyreader", "pilreader", "nibabelreader"],
            )

    def test_get_default_reader_registration_order_nvimgcodec_env(self):
        with patch.dict(os.environ, {"MONAI_DICOM_READER": "nvimgcodec"}):
            order = get_default_reader_registration_order()
            self.assertEqual(order[-1], "nvimgcodecpydicomreader")
            self.assertIn("pydicomreader", order)
            self.assertIn("itkreader", order)

    def test_custom_supported_reader_included_in_order(self):
        class _DummyReader:
            pass

        with patch.dict(SUPPORTED_READERS, {"dummyreader": _DummyReader}):
            with patch.dict(os.environ, {"MONAI_DICOM_READER": "itk"}):
                order = get_default_reader_registration_order()
                self.assertIn("dummyreader", order)
                self.assertEqual(order[-1], "itkreader")
                self.assertLess(order.index("dummyreader"), order.index("itkreader"))

    def test_dicom_reader_env_map_values(self):
        self.assertEqual(set(DICOM_READER_ENV_MAP.keys()), {"itk", "pydicom", "nvimgcodec"})


class TestNvImgCodecPydicomReader(unittest.TestCase):
    @SkipIfNoModule("pydicom")
    @patch("monai.data.nvimgcodec_pydicom_plugin.register_as_decoder_plugin", return_value=True)
    @patch("monai.data.nvimgcodec_pydicom_plugin.is_nvimgcodec_available", return_value=True)
    def test_reader_init_registers_plugin(self, _mock_available, mock_register):
        from monai.data import NvImgCodecPydicomReader

        reader = NvImgCodecPydicomReader()
        self.assertIsInstance(reader, NvImgCodecPydicomReader)
        mock_register.assert_called_once()

    @SkipIfNoModule("pydicom")
    @patch("monai.data.nvimgcodec_pydicom_plugin.is_nvimgcodec_available", return_value=False)
    def test_verify_suffix_without_nvimgcodec(self, _mock_available):
        from monai.data import NvImgCodecPydicomReader

        reader = NvImgCodecPydicomReader()
        self.assertFalse(reader.verify_suffix("tests/testing_data/CT_DICOM"))

    @SkipIfNoModule("pydicom")
    @patch("monai.data.nvimgcodec_pydicom_plugin.is_nvimgcodec_available", return_value=True)
    def test_verify_suffix_with_nvimgcodec(self, _mock_available):
        from monai.data import NvImgCodecPydicomReader

        reader = NvImgCodecPydicomReader()
        self.assertTrue(reader.verify_suffix("tests/testing_data/CT_DICOM"))
        self.assertFalse(reader.verify_suffix("tests/testing_data/test_image.nii.gz"))

    @SkipIfNoModule("pydicom")
    @patch("monai.data.nvimgcodec_pydicom_plugin.register_as_decoder_plugin", return_value=True)
    @patch("monai.data.nvimgcodec_pydicom_plugin.is_nvimgcodec_available", return_value=True)
    def test_to_gpu_ignored(self, _mock_available, _mock_register):
        from monai.data import NvImgCodecPydicomReader

        with self.assertWarns(UserWarning) as warning_ctx:
            reader = NvImgCodecPydicomReader(to_gpu=True)
        self.assertFalse(reader.to_gpu)
        self.assertIn("ignores to_gpu=True", str(warning_ctx.warning))

        reader = NvImgCodecPydicomReader(to_gpu=False)
        self.assertFalse(reader.to_gpu)


class TestLoadImageDicomReaderEnv(unittest.TestCase):
    @SkipIfNoModule("pydicom")
    def test_load_image_respects_dicom_reader_env(self):
        with patch.dict(os.environ, {"MONAI_DICOM_READER": "pydicom"}):
            loader = LoadImage(image_only=True)
            reader_types = [type(r).__name__ for r in loader.readers]
            self.assertEqual(reader_types[-1], "PydicomReader")
            # Other DICOM readers from SUPPORTED_READERS remain when their deps are available.
            self.assertTrue(any(name != "PydicomReader" for name in reader_types))

    @SkipIfNoModule("pydicom")
    @patch("monai.data.nvimgcodec_pydicom_plugin.is_nvimgcodec_available", return_value=True)
    @patch("monai.data.nvimgcodec_pydicom_plugin.register_as_decoder_plugin", return_value=True)
    def test_load_image_nvimgcodec_env(self, _mock_register, _mock_available):
        with patch.dict(os.environ, {"MONAI_DICOM_READER": "nvimgcodec"}):
            loader = LoadImage(image_only=True)
            reader_types = [type(r).__name__ for r in loader.readers]
            self.assertEqual(reader_types[-1], "NvImgCodecPydicomReader")

    @SkipIfNoModule("pydicom")
    @patch("monai.data.nvimgcodec_pydicom_plugin.register_as_decoder_plugin", return_value=False)
    @patch("monai.data.nvimgcodec_pydicom_plugin.is_nvimgcodec_available", return_value=False)
    def test_load_image_nvimgcodec_env_unavailable_auto_select(self, _mock_available, _mock_register):
        """When nvimgcodec is unavailable, verify_suffix skips it and another DICOM reader can load."""
        with patch.dict(os.environ, {"MONAI_DICOM_READER": "nvimgcodec"}):
            order = get_default_reader_registration_order()
            self.assertEqual(order[-1], "nvimgcodecpydicomreader")

            loader = LoadImage(image_only=True)
            reader_types = [type(r).__name__ for r in loader.readers]
            self.assertEqual(reader_types[-1], "NvImgCodecPydicomReader")
            self.assertIn("PydicomReader", reader_types)

            nv_reader = loader.readers[-1]
            self.assertFalse(nv_reader.verify_suffix("tests/testing_data/CT_DICOM"))

            # Preferred reader is skipped; a later DICOM reader in the list should succeed.
            img = loader("tests/testing_data/CT_DICOM")
            self.assertIsNotNone(img)


class TestNvImgCodecPluginRegistration(unittest.TestCase):
    @SkipIfNoModule("pydicom")
    @SkipIfNoModule("nvidia.nvimgcodec.tools.dicom.pydicom_plugin")
    @patch("monai.data.nvimgcodec_pydicom_plugin.is_nvimgcodec_available", return_value=True)
    def test_register_as_decoder_plugin(self, _mock_available):
        from pydicom.pixels.decoders import JPEGBaseline8BitDecoder

        from monai.data.nvimgcodec_pydicom_plugin import (
            NVIMGCODEC_PLUGIN_LABEL,
            register_as_decoder_plugin,
            unregister_as_decoder_plugin,
        )

        self.assertTrue(register_as_decoder_plugin())
        self.assertIn(NVIMGCODEC_PLUGIN_LABEL, JPEGBaseline8BitDecoder.available_plugins)
        self.assertTrue(unregister_as_decoder_plugin())
        self.assertNotIn(NVIMGCODEC_PLUGIN_LABEL, JPEGBaseline8BitDecoder.available_plugins)

    @SkipIfNoModule("pydicom")
    @patch("monai.data.nvimgcodec_pydicom_plugin.is_nvimgcodec_available", return_value=False)
    def test_register_without_nvimgcodec(self, _mock_available):
        from monai.data.nvimgcodec_pydicom_plugin import register_as_decoder_plugin

        self.assertFalse(register_as_decoder_plugin())

    @SkipIfNoModule("pydicom")
    @SkipIfNoModule("nvidia.nvimgcodec.tools.dicom.pydicom_plugin")
    def test_is_nvimgcodec_available_with_cuda(self):
        from monai.data.nvimgcodec_pydicom_plugin import is_nvimgcodec_available

        # When CUDA and nvimgcodec are present this should be True; otherwise skip-like behavior.
        if is_nvimgcodec_available():
            from monai.data.nvimgcodec_pydicom_plugin import SUPPORTED_TRANSFER_SYNTAXES, is_available

            self.assertTrue(is_available(SUPPORTED_TRANSFER_SYNTAXES[0]))


class TestNvImgCodecPydicomReaderIntegration(unittest.TestCase):
    @SkipIfNoModule("pydicom")
    @SkipIfNoModule("nvidia.nvimgcodec.tools.dicom.pydicom_plugin")
    def test_load_dicom_with_pydicom_env(self):
        with patch.dict(os.environ, {"MONAI_DICOM_READER": "pydicom"}):
            result = LoadImage(image_only=True)("tests/testing_data/CT_DICOM")
            self.assertEqual(tuple(result.shape), (16, 16, 4))

    @SkipIfNoModule("pydicom")
    @SkipIfNoModule("nvidia.nvimgcodec.tools.dicom.pydicom_plugin")
    @patch("monai.data.nvimgcodec_pydicom_plugin.register_as_decoder_plugin", return_value=False)
    @patch("monai.data.nvimgcodec_pydicom_plugin.is_nvimgcodec_available", return_value=False)
    def test_load_dicom_with_nvimgcodec_reader_fallback(self, _mock_available, _mock_register):
        from monai.data import NvImgCodecPydicomReader

        reader = NvImgCodecPydicomReader()
        result = LoadImage(image_only=True, reader=reader)("tests/testing_data/CT_DICOM")
        self.assertEqual(tuple(result.shape), (16, 16, 4))


if __name__ == "__main__":
    unittest.main()

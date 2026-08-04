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

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

from monai.data.wsi_reader import WSIReader
from monai.utils import optional_import
from tests.test_utils import SkipIfNoModule


def _optional_import_with_wsidicom(name=None, module=None, **kwargs):
    if module == "wsidicom" or name == "WsiDicom":
        return MagicMock(), True
    return optional_import(module=module, name=name, **kwargs)


class TestWsiDicomWSIReader(unittest.TestCase):
    @patch("monai.data.wsi_reader.optional_import")
    @patch("monai.utils.module.optional_import", side_effect=_optional_import_with_wsidicom)
    def test_verify_suffix_uses_wsidicom_is_supported(self, _mock_module_optional, mock_optional_import):
        mock_wsi_cls = MagicMock()
        mock_wsi_cls.is_supported.side_effect = lambda path: str(path).endswith("dicom_wsi")
        mock_optional_import.return_value = (mock_wsi_cls, True)

        from monai.data import WsiDicomWSIReader

        reader = WsiDicomWSIReader(register_nvimgcodec=False, prefer_pydicom_decoder=False)
        self.assertTrue(reader.verify_suffix("/data/slide.dicom_wsi"))
        self.assertFalse(reader.verify_suffix("/data/slide.svs"))

    @patch("monai.data.nvimgcodec_pydicom_plugin.configure_wsidicom_pydicom_decoder")
    @patch("monai.data.wsi_reader.optional_import")
    @patch("monai.utils.module.optional_import", side_effect=_optional_import_with_wsidicom)
    def test_init_registers_nvimgcodec_by_default(self, _mock_module_optional, mock_optional_import, mock_configure):
        mock_optional_import.return_value = (MagicMock(), True)
        from monai.data import WsiDicomWSIReader

        WsiDicomWSIReader()
        mock_configure.assert_called_once_with(register_nvimgcodec=True, prefer_pydicom_decoder=True)

    @patch("monai.data.nvimgcodec_pydicom_plugin.configure_wsidicom_pydicom_decoder")
    @patch("monai.data.wsi_reader.optional_import")
    @patch("monai.utils.module.optional_import", side_effect=_optional_import_with_wsidicom)
    def test_init_can_skip_nvimgcodec_registration(self, _mock_module_optional, mock_optional_import, mock_configure):
        mock_optional_import.return_value = (MagicMock(), True)
        from monai.data import WsiDicomWSIReader

        WsiDicomWSIReader(register_nvimgcodec=False, prefer_pydicom_decoder=False)
        mock_configure.assert_not_called()

    def _configure_with_settings(self, settings):
        """Run ``configure_wsidicom_pydicom_decoder`` against a stand-in wsidicom settings."""
        with (
            patch(
                "monai.data.nvimgcodec_pydicom_plugin.register_as_decoder_plugin", return_value=True
            ) as mock_register_plugin,
            patch("monai.data.nvimgcodec_pydicom_plugin.register_pixels_handler_for_wsidicom") as mock_register_handler,
            patch(
                "monai.data.nvimgcodec_pydicom_plugin.optional_import",
                return_value=(SimpleNamespace(settings=settings), True),
            ),
        ):
            from monai.data.nvimgcodec_pydicom_plugin import configure_wsidicom_pydicom_decoder

            configure_wsidicom_pydicom_decoder()
            mock_register_plugin.assert_called_once()
            mock_register_handler.assert_called_once()

    def test_configure_wsidicom_prefers_pydicom_decoder(self):
        # Declared on the class, as wsidicom exposes it as a property. An instance-only
        # stand-in would accept any attribute name and hide a misspelling.
        class Settings:
            preferred_decoder = None

        settings = Settings()
        self._configure_with_settings(settings)
        self.assertEqual(settings.preferred_decoder, "pydicom")

    def test_configure_wsidicom_supports_legacy_decoder_setting_name(self):
        class LegacySettings:
            prefered_decoder = None

        settings = LegacySettings()
        self._configure_with_settings(settings)
        self.assertEqual(settings.prefered_decoder, "pydicom")

    def test_configure_wsidicom_does_not_invent_decoder_setting(self):
        class UnknownSettings:
            pass

        settings = UnknownSettings()
        with self.assertLogs("monai.data.nvimgcodec_pydicom_plugin", level="WARNING"):
            self._configure_with_settings(settings)
        self.assertEqual(vars(settings), {})

    @patch("monai.data.wsi_reader.optional_import")
    @patch("monai.utils.module.optional_import", side_effect=_optional_import_with_wsidicom)
    def test_get_patch_converts_level0_location_to_wsidicom(self, _mock_module_optional, mock_optional_import):
        mock_optional_import.return_value = (MagicMock(), True)

        from monai.data import WsiDicomWSIReader

        reader = WsiDicomWSIReader(register_nvimgcodec=False, prefer_pydicom_decoder=False)
        wsi = MagicMock()
        level_obj = SimpleNamespace(size=SimpleNamespace(height=1000, width=2000))
        base_level = SimpleNamespace(size=SimpleNamespace(height=4000, width=8000))
        wsi.pyramid.get.side_effect = lambda level, pyramid_index=True: level_obj if level == 1 else base_level
        wsi.pyramid.highest_level = 2
        wsi.mpp = SimpleNamespace(height=0.25, width=0.25)
        wsi.read_region.return_value = Image.new("RGB", (128, 64), color=(1, 2, 3))

        patch = reader._get_patch(wsi, location=(200, 400), size=(64, 128), level=1, dtype=np.uint8, mode="RGB")

        wsi.read_region.assert_called_once_with((100, 50), 1, (128, 64), threads=1)
        self.assertEqual(patch.shape, (3, 64, 128))

    @patch("monai.data.wsi_reader.WsiDicomWSIReader", autospec=True)
    def test_wsi_reader_backend_dispatch(self, mock_reader_cls):
        mock_reader_cls.return_value.supported_suffixes = []
        mock_reader_cls.return_value.level = 0
        mock_reader_cls.return_value.mpp_rtol = 0.05
        mock_reader_cls.return_value.mpp_atol = 0.0
        mock_reader_cls.return_value.power = None
        mock_reader_cls.return_value.power_rtol = 0.05
        mock_reader_cls.return_value.power_atol = 0.0
        mock_reader_cls.return_value.channel_dim = 0
        mock_reader_cls.return_value.dtype = np.uint8
        mock_reader_cls.return_value.device = None
        mock_reader_cls.return_value.mode = "RGB"
        mock_reader_cls.return_value.kwargs = {}
        mock_reader_cls.return_value.metadata = {}
        mock_reader_cls.return_value.mpp = None

        WSIReader(backend="wsidicom")
        mock_reader_cls.assert_called_once()

    @SkipIfNoModule("wsidicom")
    def test_get_power_not_supported(self):
        from monai.data import WsiDicomWSIReader

        reader = WsiDicomWSIReader(register_nvimgcodec=False, prefer_pydicom_decoder=False)
        with self.assertRaises(ValueError):
            reader.get_power(MagicMock(), level=0)

    def test_pydicom_pixels_handler_supports_jpeg_baseline(self):
        from pydicom.uid import JPEGBaseline8Bit

        from monai.data import pydicom_pixels_handler

        self.assertTrue(pydicom_pixels_handler.supports_transfer_syntax(JPEGBaseline8Bit))
        self.assertTrue(pydicom_pixels_handler.is_available())


if __name__ == "__main__":
    unittest.main()

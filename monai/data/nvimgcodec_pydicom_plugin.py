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

"""
MONAI integration helpers for the nvImageCodec pydicom decoder plugin.

The decoder implementation lives in ``nvidia.nvimgcodec.tools.dicom.pydicom_plugin``
(shipped with ``nvidia-nvimgcodec-cuXX``). This module provides MONAI-facing helpers
and stable aliases for registration and availability checks.
"""

from __future__ import annotations

import logging
from typing import Optional

from monai.utils import optional_import

cp, has_cp = optional_import("cupy")
pydicom_plugin, has_pydicom_plugin = optional_import("nvidia.nvimgcodec.tools.dicom.pydicom_plugin")

_logger = logging.getLogger(__name__)

if has_pydicom_plugin:
    DECODER_DEPENDENCIES = pydicom_plugin.DECODER_DEPENDENCIES
    NVIMGCODEC_MIN_VERSION = pydicom_plugin.NVIMGCODEC_MIN_VERSION
    NVIMGCODEC_MIN_VERSION_TUPLE = pydicom_plugin.NVIMGCODEC_MIN_VERSION_TUPLE
    NVIMGCODEC_PLUGIN_LABEL = pydicom_plugin.NVIMGCODEC_PLUGIN_LABEL
    SUPPORTED_DECODER_CLASSES = pydicom_plugin.SUPPORTED_DECODER_CLASSES
    SUPPORTED_TRANSFER_SYNTAXES = pydicom_plugin.SUPPORTED_TRANSFER_SYNTAXES
    is_available = pydicom_plugin.is_available
else:  # pragma: no cover - optional dependency not installed
    DECODER_DEPENDENCIES = {}
    NVIMGCODEC_MIN_VERSION = "0.8.0"
    NVIMGCODEC_MIN_VERSION_TUPLE = (0, 8, 0)
    NVIMGCODEC_PLUGIN_LABEL = "0.8.0+nvimgcodec"
    SUPPORTED_DECODER_CLASSES = []
    SUPPORTED_TRANSFER_SYNTAXES = []

    def is_available(uid) -> bool:  # type: ignore[no-redef]
        return False


def is_nvimgcodec_available() -> bool:
    """Return ``True`` if nvImageCodec with CUDA support is available."""
    if not has_pydicom_plugin or getattr(pydicom_plugin, "nvimgcodec", None) is None or not has_cp:
        _logger.debug("nvimgcodec pydicom plugin, nvimgcodec module, or CuPy missing.")
        return False
    try:
        if not cp.cuda.is_available():
            _logger.debug("CUDA device not found.")
            return False
    except Exception as exc:  # pragma: no cover - environment specific
        _logger.debug(f"CUDA availability check failed: {exc}")
        return False
    return True


def register_as_decoder_plugin(module_path: str | None = None) -> bool:
    """Register the nvImageCodec pydicom decoder plugin."""
    if not is_nvimgcodec_available():
        _logger.warning("nvImageCodec is not available; skipping pydicom decoder plugin registration.")
        return False
    if not has_pydicom_plugin:
        return False
    return pydicom_plugin.register(module_path)


def unregister_as_decoder_plugin() -> bool:
    """Unregister the nvImageCodec pydicom decoder plugin."""
    if not has_pydicom_plugin:
        return False
    return pydicom_plugin.unregister()

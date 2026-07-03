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
Legacy pydicom ``pixel_data_handlers`` bridge to the pydicom 3 ``pixels`` backend.

wsidicom's ``PydicomDecoder`` uses the deprecated ``pydicom.config.pixel_data_handlers``
API, while nvImageCodec registers with pydicom 3 decoder plugins. This handler delegates
decoding to ``pydicom.pixels.pixel_array`` so wsidicom can use the same plugin stack,
including nvImageCodec when registered.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pydicom.dataset import Dataset
    from pydicom.uid import UID

HANDLER_NAME = "pydicom_pixels"
DEPENDENCIES = {"numpy": ("https://numpy.org/", "NumPy")}


def is_available() -> bool:
    try:
        import numpy  # noqa: F401
        from pydicom.pixels import pixel_array  # noqa: F401

        return True
    except ImportError:
        return False


def supports_transfer_syntax(transfer_syntax: UID) -> bool:
    from pydicom.pixels.decoders.base import get_decoder

    try:
        get_decoder(transfer_syntax)
    except NotImplementedError:
        return False
    return True


def needs_to_convert_to_RGB(ds: Dataset) -> bool:
    return False


def get_pixeldata(ds: Dataset):
    from pydicom.pixels import pixel_array

    return pixel_array(ds).ravel()

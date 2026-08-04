#!/usr/bin/env python
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
Diagnostic for the nvImageCodec DICOM decode path used by ``WsiDicomWSIReader``.

Reports whether the GPU decoder is importable, whether it registers with pydicom, and
whether it is actually invoked while reading a tile. Frame decode calls are counted by
wrapping the plugin's ``_decode_frame``, so a zero count means tiles were decoded by a
CPU plugin even when registration succeeded.

Example:
  python examples/pathology/check_nvimgcodec_usage.py --path /path/to/dicom_wsi_folder
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from _nvimgcodec_probe import NvImgCodecProbe, add_log_level_argument, configure_logging, gpu_memory_report


def _dependency_report() -> dict:
    """Check each dependency separately so a single missing piece is identifiable."""
    report: dict = {}

    try:
        import pydicom

        report["pydicom"] = pydicom.__version__
    except Exception as exc:
        report["pydicom"] = f"unavailable ({exc})"

    try:
        import wsidicom

        report["wsidicom"] = getattr(wsidicom, "__version__", "unknown")
    except Exception as exc:
        report["wsidicom"] = f"unavailable ({exc})"

    try:
        import cupy

        report["cupy"] = cupy.__version__
        report["cupy_cuda_available"] = bool(cupy.cuda.is_available())
    except Exception as exc:
        # A broken/partial CuPy install silently disables the whole GPU path.
        report["cupy"] = f"unavailable ({exc})"
        report["cupy_cuda_available"] = False

    try:
        from nvidia import nvimgcodec

        report["nvimgcodec"] = getattr(nvimgcodec, "__version__", "unknown")
    except Exception as exc:
        report["nvimgcodec"] = f"unavailable ({exc})"

    try:
        import nvidia.nvimgcodec.tools.dicom.pydicom_plugin as plugin

        report["pydicom_plugin"] = plugin.__file__
    except Exception as exc:
        report["pydicom_plugin"] = f"unavailable ({exc})"

    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Diagnose whether nvImageCodec decodes DICOM WSI frames.")
    parser.add_argument("--path", required=True, help="DICOM WSI folder or file.")
    parser.add_argument("--level", type=int, default=0, help="Pyramid level to read.")
    parser.add_argument("--size", type=int, nargs=2, default=[512, 512], metavar=("H", "W"), help="Tile size to read.")
    add_log_level_argument(parser)
    args = parser.parse_args(argv)

    configure_logging(args.log_level)

    result: dict = {"dependencies": _dependency_report()}

    from monai.data.nvimgcodec_pydicom_plugin import configure_wsidicom_pydicom_decoder, is_nvimgcodec_available

    result["is_nvimgcodec_available"] = is_nvimgcodec_available()
    configure_wsidicom_pydicom_decoder(register_nvimgcodec=True, prefer_pydicom_decoder=True)

    import nvidia.nvimgcodec.tools.dicom.pydicom_plugin as plugin

    label = plugin.NVIMGCODEC_PLUGIN_LABEL
    registered = []
    for decoder in plugin.SUPPORTED_DECODER_CLASSES:
        plugins = getattr(decoder, "_available", {}) or {}
        if label in plugins:
            registered.append(getattr(decoder, "UID", str(decoder)))
    result["registered_decoders"] = [str(uid) for uid in registered]

    # Registration alone does not prove use, and pydicom falls back to a CPU plugin when
    # this one raises, so successes must be counted separately from failures.
    probe = NvImgCodecProbe().start()

    from monai.data import WSIReader

    reader = WSIReader(backend="wsidicom", register_nvimgcodec=True, prefer_pydicom_decoder=True)
    wsi = reader.read(args.path)
    if isinstance(wsi, (list, tuple)):
        wsi = wsi[0]

    patch, _ = reader.get_data(wsi, location=(0, 0), size=(args.size[0], args.size[1]), level=args.level)
    probe.stop()

    result["tile_shape"] = list(patch.shape)
    result["nvimgcodec"] = probe.report()
    if probe.successes:
        result["gpu_memory"] = gpu_memory_report()

    print(json.dumps(result, indent=2))
    return 0 if probe.successes else 2


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    raise SystemExit(main())

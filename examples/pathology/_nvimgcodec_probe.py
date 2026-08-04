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
Shared instrumentation for the nvImageCodec DICOM decode path.

pydicom tries each registered decoder plugin in turn and, on exception, logs it and moves
to the next plugin, only raising when every plugin fails. A plugin can therefore be called
for every frame while contributing no decoded pixels, so counting calls alone cannot show
whether GPU decoding worked. The probe here separates successes from failures and records
the first error.
"""

from __future__ import annotations

import logging

LOG_LEVELS = ("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG")

# Emitting decode failures at their original ERROR level keeps them visible without
# requiring --log-level, since a silent CPU fallback is the failure mode being hunted.
_PYDICOM_DECODE_LOGGER = "pydicom.pixels"


def add_log_level_argument(parser) -> None:
    """Add a ``--log-level`` option controlling decoder plugin verbosity."""
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=LOG_LEVELS,
        help="Logging level. Use DEBUG to trace decoder plugin selection and decode calls.",
    )


def configure_logging(level_name: str) -> None:
    """Configure root logging and make decoder plugin loggers follow ``level_name``."""
    level = getattr(logging, level_name.upper())
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger().setLevel(level)
    for name in ("nvidia.nvimgcodec", _PYDICOM_DECODE_LOGGER, "monai.data.nvimgcodec_pydicom_plugin", "wsidicom"):
        logging.getLogger(name).setLevel(level)


class NvImgCodecProbe:
    """Counts successful and failed nvImageCodec frame decodes."""

    def __init__(self) -> None:
        self.available = False
        self.successes = 0
        self.failures = 0
        self.first_error: str | None = None
        self._plugin = None
        self._original_decode_frame = None

    def start(self) -> NvImgCodecProbe:
        """Wrap the plugin's frame decoder. Safe to call when the plugin is absent."""
        try:
            import nvidia.nvimgcodec.tools.dicom.pydicom_plugin as plugin
        except Exception:
            return self

        self._plugin = plugin
        self._original_decode_frame = plugin._decode_frame
        original = plugin._decode_frame

        def counting_decode_frame(*args, **kwargs):
            try:
                frame = original(*args, **kwargs)
            except Exception as exc:
                self.failures += 1
                if self.first_error is None:
                    self.first_error = f"{type(exc).__name__}: {exc}"
                raise
            self.successes += 1
            return frame

        plugin._decode_frame = counting_decode_frame
        # pydicom stores the resolved callable per decoder, so patching the module
        # attribute alone would not intercept calls.
        label = plugin.NVIMGCODEC_PLUGIN_LABEL
        for decoder in plugin.SUPPORTED_DECODER_CLASSES:
            available = getattr(decoder, "_available", None)
            if available is not None and label in available:
                available[label] = counting_decode_frame
        self.available = True
        return self

    def stop(self) -> None:
        """Restore the original frame decoder."""
        if self._plugin is None or self._original_decode_frame is None:
            return
        plugin = self._plugin
        plugin._decode_frame = self._original_decode_frame
        label = plugin.NVIMGCODEC_PLUGIN_LABEL
        for decoder in plugin.SUPPORTED_DECODER_CLASSES:
            available = getattr(decoder, "_available", None)
            if available is not None and label in available:
                available[label] = self._original_decode_frame

    def report(self) -> dict:
        return {
            "plugin_instrumented": self.available,
            "frame_decodes_succeeded": self.successes,
            "frame_decodes_failed": self.failures,
            "first_error": self.first_error,
            # Only successful decodes produced GPU pixels; failures fall back to CPU.
            "nvimgcodec_used": self.successes > 0,
        }


def gpu_memory_report() -> dict:
    """
    Report GPU memory for this process.

    ``nvidia-smi`` polls once per second and often misses a short decode run, so sample
    from inside the process instead.
    """
    report: dict = {}
    try:
        import cupy as cp
    except Exception as exc:
        return {"error": f"cupy unavailable ({exc})"}

    try:
        free, total = cp.cuda.runtime.memGetInfo()
        report["device_free_mb"] = round(free / 1024**2, 1)
        report["device_total_mb"] = round(total / 1024**2, 1)
        report["device_used_mb"] = round((total - free) / 1024**2, 1)
    except Exception as exc:
        report["error"] = f"memGetInfo failed ({exc})"

    try:
        import os

        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(cp.cuda.runtime.getDevice())
        report["device_name"] = pynvml.nvmlDeviceGetName(handle)
        pid = os.getpid()
        for proc in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
            if proc.pid == pid:
                used = proc.usedGpuMemory
                report["this_process_mb"] = round(used / 1024**2, 1) if used else None
        report.setdefault("this_process_mb", None)
        pynvml.nvmlShutdown()
    except Exception as exc:
        report["pynvml_error"] = f"{type(exc).__name__}: {exc}"

    return report

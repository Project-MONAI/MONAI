from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from monai.data import save_net_with_metadata
from monai.data.torchscript_utils import save_onnx


class OnnxSaver:
    def save(self, model_obj, filepath):
        save_onnx(
            model_obj=model_obj,
            filepath=filepath
        )

class CkptSaver:
    def __init__(self, include_config_vals: bool = True, append_timestamp: bool = False, meta_values: Mapping[str, Any] | None = None, more_extra_files: Mapping[str, bytes] | None = None):
        self.include_config_vals = include_config_vals
        self.append_timestamp = append_timestamp
        self.meta_values = meta_values
        self.more_extra_files = more_extra_files

    def save(self, model_obj, filepath):
        save_net_with_metadata(
            model_obj=model_obj,
            filepath=filepath,
            include_config_vals=self.include_config_vals,
            append_timestamp=self.append_timestamp,
            meta_values=self.meta_values,
            more_extra_files=self.more_extra_files
        )

class TrtSaver:
    def __init__(self, include_config_vals: bool = True, append_timestamp: bool = False, meta_values: Mapping[str, Any] | None = None, more_extra_files: Mapping[str, bytes] | None = None):
        self.include_config_vals = include_config_vals
        self.append_timestamp = append_timestamp
        self.meta_values = meta_values
        self.more_extra_files = more_extra_files

    def save(self, model_obj, filepath):
        save_net_with_metadata(
            model_obj=model_obj,
            filepath=filepath,
            include_config_vals=self.include_config_vals,
            append_timestamp=self.append_timestamp,
            meta_values=self.meta_values,
            more_extra_files=self.more_extra_files
        )

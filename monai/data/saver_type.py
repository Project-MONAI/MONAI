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

from collections.abc import Mapping
from typing import IO, Any

from monai.data import save_net_with_metadata
from monai.utils.module import optional_import

onnx, _ = optional_import("onnx")


def save_onnx(model_obj: Any, filepath: str | IO[Any]) -> None:
    """
    Save the ONNX model to the given file or stream.

    Args:
        model_obj: ONNX model to save.
        filepath: Filename or file-like stream object to save the ONNX model.
    """
    onnx.save(model_obj, filepath)


class OnnxSaver:
    def save(self, model_obj, filepath):
        save_onnx(model_obj=model_obj, filepath=filepath)


class CkptSaver:
    def __init__(
        self,
        include_config_vals: bool = True,
        append_timestamp: bool = False,
        meta_values: Mapping[str, Any] | None = None,
        more_extra_files: Mapping[str, bytes] | None = None,
    ):
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
            more_extra_files=self.more_extra_files,
        )


class TrtSaver:
    def __init__(
        self,
        include_config_vals: bool = True,
        append_timestamp: bool = False,
        meta_values: Mapping[str, Any] | None = None,
        more_extra_files: Mapping[str, bytes] | None = None,
    ):
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
            more_extra_files=self.more_extra_files,
        )

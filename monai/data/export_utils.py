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

import datetime
import json
import os
from collections.abc import Mapping, Sequence
from typing import IO, Any

import torch

from monai.config import get_config_values
from monai.data.torchscript_utils import METADATA_FILENAME
from monai.utils import ExportMetadataKeys

__all__ = ["load_exported_program", "save_exported_program"]


def save_exported_program(
    exported_program: torch.export.ExportedProgram,
    filename_prefix_or_stream: str | os.PathLike | IO[bytes],
    include_config_vals: bool = True,
    append_timestamp: bool = False,
    meta_values: Mapping[str, Any] | None = None,
    more_extra_files: Mapping[str, Any] | None = None,
) -> None:
    """
    Save an ``ExportedProgram`` produced by :func:`torch.export.export` with metadata included
    as a JSON file inside the ``.pt2`` archive.

    Examples::

        import torch
        from monai.networks.nets import UNet

        net = UNet(spatial_dims=2, in_channels=1, out_channels=1, channels=[8, 16], strides=[2])
        ep = torch.export.export(net, args=(torch.rand(1, 1, 32, 32),))

        meta = {"name": "Test UNet", "input_dims": 2}
        save_exported_program(ep, "test", meta_values=meta)

        loaded_ep, loaded_meta, _ = load_exported_program("test.pt2")

    Args:
        exported_program: an ``ExportedProgram`` returned by :func:`torch.export.export`.
        filename_prefix_or_stream: filename or file-like stream object.
            If a string filename has no extension it becomes ``.pt2``.
        include_config_vals: if True, MONAI, PyTorch, and NumPy versions are included in metadata.
        append_timestamp: if True, a timestamp is appended to the filename before the extension.
        meta_values: metadata values to store, compatible with JSON serialization.
        more_extra_files: additional data items to include in the archive.
    """
    now = datetime.datetime.now()
    metadict: dict[str, Any] = {}

    if include_config_vals:
        metadict.update(get_config_values())
        metadict[ExportMetadataKeys.TIMESTAMP.value] = now.astimezone().isoformat()

    if meta_values is not None:
        metadict.update(meta_values)

    json_data = json.dumps(metadict)

    extra_files: dict[str, Any] = {METADATA_FILENAME: json_data}

    if more_extra_files is not None:
        if METADATA_FILENAME in more_extra_files:
            raise ValueError(f"'{METADATA_FILENAME}' is reserved and cannot be used in more_extra_files.")
        extra_files.update(more_extra_files)

    # torch.export.save requires str values; decode bytes from legacy callers (e.g. _export helper)
    extra_files = {k: v.decode() if isinstance(v, bytes) else v for k, v in extra_files.items()}

    if isinstance(filename_prefix_or_stream, (str, os.PathLike)):
        filename_prefix_or_stream = str(filename_prefix_or_stream)
        filename_no_ext, ext = os.path.splitext(filename_prefix_or_stream)
        if ext == "":
            ext = ".pt2"

        if append_timestamp:
            filename_prefix_or_stream = now.strftime(f"{filename_no_ext}_%Y%m%d%H%M%S{ext}")
        else:
            filename_prefix_or_stream = filename_no_ext + ext

    torch.export.save(exported_program, filename_prefix_or_stream, extra_files=extra_files)


def load_exported_program(
    filename_prefix_or_stream: str | os.PathLike | IO[bytes], more_extra_files: Sequence[str] = ()
) -> tuple[torch.export.ExportedProgram, dict, dict]:
    """
    Load an ``ExportedProgram`` from a ``.pt2`` file and extract stored JSON metadata.

    Args:
        filename_prefix_or_stream: filename or file-like stream object.
        more_extra_files: additional extra file names to load from the archive.

    Returns:
        Triple of (ExportedProgram, metadata dict, extra files dict).
    """
    extra_files: dict[str, Any] = dict.fromkeys(more_extra_files, "")
    extra_files[METADATA_FILENAME] = ""

    exported_program = torch.export.load(filename_prefix_or_stream, extra_files=extra_files)
    json_data_dict = json.loads(extra_files.pop(METADATA_FILENAME))

    return exported_program, json_data_dict, extra_files

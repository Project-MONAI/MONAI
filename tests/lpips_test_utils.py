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

import hashlib
import shutil
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from urllib.parse import urlparse

import torch

from monai.networks.nets.perceptual import LPIPS_NETWORKS


@contextmanager
def lpips_test_cache(preload: bool = True):
    """Provide synthetic calibration files while retaining real cache/hash/load behavior.

    Args:
        preload: Whether the checkpoint cache is initially populated.

    Yields:
        The cache directory, expected calibration tensors and mocked network transport.
    """
    channels = {
        "alex": (64, 192, 384, 256, 256),
        "vgg": (64, 128, 256, 512, 512),
        "squeeze": (64, 128, 256, 384, 384, 512, 512),
    }
    with TemporaryDirectory() as directory:
        root = Path(directory)
        cache = root / "hub"
        expected = {}
        sources = {}
        metadata = {}
        for net, sizes in channels.items():
            expected[net] = {
                f"lin{i}.model.1.weight": torch.full((1, size, 1, 1), 0.125) for i, size in enumerate(sizes)
            }
            sources[net] = root / f"{net}.pth"
            torch.save(expected[net], sources[net])
            digest = hashlib.sha256(sources[net].read_bytes()).hexdigest()
            metadata[net] = (*LPIPS_NETWORKS[net][:-1], digest)
            if preload:
                destination = cache / "checkpoints" / f"lpips_v0.1_{net}.pth"
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(sources[net], destination)

        def fetch(url, filepath, progress):
            """Copy a synthetic source in place of the external HTTP transfer.

            Args:
                url: Official checkpoint URL identifying the network.
                filepath: Temporary destination supplied by download_url.
                progress: Requested progress display, unused for the local copy.
            """
            net = Path(urlparse(url).path).stem
            shutil.copyfile(sources[net], filepath)

        with (
            patch.dict(LPIPS_NETWORKS, metadata),
            patch("torch.hub.get_dir", return_value=str(cache)),
            patch("monai.apps.utils._download_with_progress", side_effect=fetch) as transfer,
        ):
            yield cache, expected, transfer

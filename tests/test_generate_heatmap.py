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

import numpy as np
import pytest
import torch

from monai.data import MetaTensor
from monai.transforms.post.array import GenerateHeatmap
from monai.transforms.post.dictionary import GenerateHeatmapd


def test_generate_heatmap_array_2d() -> None:
    points = np.array([[4.2, 7.8], [12.3, 3.6]], dtype=np.float32)
    transform = GenerateHeatmap(sigma=1.5, spatial_shape=(16, 16))

    heatmap = transform(points)

    assert heatmap.shape == (2, 16, 16)
    assert heatmap.dtype == np.float32
    np.testing.assert_allclose(heatmap.max(axis=(1, 2)), np.ones(2), rtol=1e-5, atol=1e-5)

    for idx, channel in enumerate(heatmap):
        max_idx = np.array(np.unravel_index(np.argmax(channel), channel.shape))
        assert np.all(np.abs(max_idx - points[idx]) <= 1)
        assert channel[0, 0] < 1e-3


def test_generate_heatmap_array_torch_output() -> None:
    points = torch.tensor([[1.5, 2.5, 3.5]], dtype=torch.float32)
    transform = GenerateHeatmap(sigma=1.0, spatial_shape=(8, 8, 8), dtype=torch.float32)

    heatmap = transform(points.to(device=points.device))

    assert isinstance(heatmap, torch.Tensor)
    assert heatmap.device == points.device
    assert heatmap.shape == (1, 8, 8, 8)
    assert torch.isclose(heatmap.max(), torch.tensor(1.0, dtype=heatmap.dtype, device=heatmap.device))


def test_generate_heatmapd_with_reference_meta() -> None:
    points = np.array([[2.5, 2.5, 3.0], [5.0, 5.0, 4.0]], dtype=np.float32)
    affine = torch.eye(4)
    image = MetaTensor(torch.zeros((1, 8, 8, 8), dtype=torch.float32), affine=affine)
    image.meta["spatial_shape"] = (8, 8, 8)
    data = {"points": points, "image": image}

    transform = GenerateHeatmapd(
        keys="points",
        heatmap_keys="heatmap",
        ref_image_keys="image",
        sigma=2.0,
    )

    result = transform(data)
    heatmap = result["heatmap"]

    assert isinstance(heatmap, MetaTensor)
    assert tuple(heatmap.shape) == (2, 8, 8, 8)
    assert heatmap.meta["spatial_shape"] == (8, 8, 8)
    assert torch.allclose(heatmap.affine, image.affine)
    np.testing.assert_allclose(heatmap.cpu().numpy().max(axis=(1, 2, 3)), np.ones(2), rtol=1e-5, atol=1e-5)


def test_generate_heatmapd_static_shape() -> None:
    points = np.array([[1.0, 1.0]], dtype=np.float32)
    transform = GenerateHeatmapd(keys="points", heatmap_keys="heatmap", spatial_shape=(6, 6))

    result = transform({"points": points})

    heatmap = result["heatmap"]
    assert isinstance(heatmap, np.ndarray)
    assert heatmap.shape == (1, 6, 6)


def test_generate_heatmapd_missing_shape_raises() -> None:
    transform = GenerateHeatmapd(keys="points", heatmap_keys="heatmap")

    with pytest.raises(ValueError):
        transform({"points": np.zeros((1, 2), dtype=np.float32)})

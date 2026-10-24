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
from typing import Any

import numpy as np
import torch
from parameterized import parameterized

from monai.networks.nets import DenseNet, DenseNet121, SEResNet50
from monai.visualize import GradCAM, GradCAMpp, ModelWithHooks
from tests.test_utils import assert_allclose, skip_if_quick


class DenseNetAdjoint(DenseNet121):
    def __call__(self, x, adjoint_info):
        if adjoint_info != 42:
            raise ValueError
        return super().__call__(x)


TESTS: list[Any] = []
TESTS_ILL: list[Any] = []

for cam in (GradCAM, GradCAMpp):
    # 2D
    TESTS.append(
        [
            cam,
            {
                "model": "densenet2d",
                "shape": (2, 1, 48, 64),
                "feature_shape": (2, 1, 1, 2),
                "target_layers": "class_layers.relu",
            },
            (2, 1, 48, 64),
        ]
    )
    # 2D binary classification (out_channels=1)
    TESTS.append(
        [
            cam,
            {
                "model": "densenet2d_bin",
                "shape": (2, 1, 48, 64),
                "feature_shape": (2, 1, 1, 2),
                "target_layers": "class_layers.relu",
            },
            (2, 1, 48, 64),
        ]
    )
    # 3D
    TESTS.append(
        [
            cam,
            {
                "model": "densenet3d",
                "shape": (2, 1, 6, 6, 6),
                "feature_shape": (2, 1, 2, 2, 2),
                "target_layers": "class_layers.relu",
            },
            (2, 1, 6, 6, 6),
        ]
    )
    # 3D binary classification (out_channels=1)
    TESTS.append(
        [
            cam,
            {
                "model": "densenet3d_bin",
                "shape": (2, 1, 6, 6, 6),
                "feature_shape": (2, 1, 2, 2, 2),
                "target_layers": "class_layers.relu",
            },
            (2, 1, 6, 6, 6),
        ]
    )
    # 2D
    TESTS.append(
        [
            cam,
            {"model": "senet2d", "shape": (2, 3, 64, 64), "feature_shape": (2, 1, 2, 2), "target_layers": "layer4"},
            (2, 1, 64, 64),
        ]
    )
    # 2D binary classification (num_classes=1)
    TESTS.append(
        [
            cam,
            {"model": "senet2d_bin", "shape": (2, 3, 64, 64), "feature_shape": (2, 1, 2, 2), "target_layers": "layer4"},
            (2, 1, 64, 64),
        ]
    )

    # 3D
    TESTS.append(
        [
            cam,
            {
                "model": "senet3d",
                "shape": (2, 3, 8, 8, 48),
                "feature_shape": (2, 1, 1, 1, 2),
                "target_layers": "layer4",
            },
            (2, 1, 8, 8, 48),
        ]
    )
    # 3D binary classification (num_classes=1)
    TESTS.append(
        [
            cam,
            {
                "model": "senet3d_bin",
                "shape": (2, 3, 8, 8, 48),
                "feature_shape": (2, 1, 1, 1, 2),
                "target_layers": "layer4",
            },
            (2, 1, 8, 8, 48),
        ]
    )

    # adjoint info
    TESTS.append(
        [
            cam,
            {
                "model": "adjoint",
                "shape": (2, 1, 48, 64),
                "feature_shape": (2, 1, 1, 2),
                "target_layers": "class_layers.relu",
            },
            (2, 1, 48, 64),
        ]
    )

    TESTS_ILL.append([cam])

TESTS_BATCH: list[Any] = [[GradCAM], [GradCAMpp]]


class TestClassScore(unittest.TestCase):
    def test_class_score(self):
        """``class_score`` returns one score per batch element for integer and per-element class indices."""
        hooks = ModelWithHooks(torch.nn.Linear(2, 4), target_layer_names=())
        logits = torch.arange(12.0).reshape(3, 4)
        assert_allclose(hooks.class_score(logits, 2), logits[:, 2])
        assert_allclose(hooks.class_score(logits, torch.tensor(2)), logits[:, 2])
        assert_allclose(hooks.class_score(logits, torch.tensor([0, 3, 1])), torch.tensor([0.0, 7.0, 9.0]))
        # one class per batch element is broadcast over the trailing (spatial) dimensions of the logits
        spatial = torch.arange(24.0).reshape(2, 3, 2, 2)
        assert_allclose(hooks.class_score(spatial, torch.tensor([2, 0])), torch.stack([spatial[0, 2], spatial[1, 0]]))
        with self.assertRaises(ValueError):
            hooks.class_score(logits, torch.tensor([0, 1]))


@skip_if_quick
class TestGradientClassActivationMap(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_shape(self, cam_class, input_data, expected_shape):
        model = None

        if input_data["model"] == "densenet2d":
            model = DenseNet121(spatial_dims=2, in_channels=1, out_channels=3)
        elif input_data["model"] == "densenet2d_bin":
            model = DenseNet(spatial_dims=2, in_channels=1, out_channels=1)
        elif input_data["model"] == "densenet3d":
            model = DenseNet(
                spatial_dims=3, in_channels=1, out_channels=3, init_features=2, growth_rate=2, block_config=(6,)
            )
        elif input_data["model"] == "densenet3d_bin":
            model = DenseNet(
                spatial_dims=3, in_channels=1, out_channels=1, init_features=2, growth_rate=2, block_config=(6,)
            )
        elif input_data["model"] == "senet2d":
            model = SEResNet50(spatial_dims=2, in_channels=3, num_classes=4)
        elif input_data["model"] == "senet2d_bin":
            model = SEResNet50(spatial_dims=2, in_channels=3, num_classes=1)
        elif input_data["model"] == "senet3d":
            model = SEResNet50(spatial_dims=3, in_channels=3, num_classes=4)
        elif input_data["model"] == "senet3d_bin":
            model = SEResNet50(spatial_dims=3, in_channels=3, num_classes=1)
        elif input_data["model"] == "adjoint":
            model = DenseNetAdjoint(spatial_dims=2, in_channels=1, out_channels=3)

        # optionally test for adjoint info
        kwargs = {"adjoint_info": 42} if input_data["model"] == "adjoint" else {}

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        cam = cam_class(nn_module=model, target_layers=input_data["target_layers"])
        image = torch.rand(input_data["shape"], device=device)
        inferred = model(image, **kwargs).max(1)[-1].cpu()
        result = cam(x=image, layer_idx=-1, **kwargs)
        np.testing.assert_array_equal(cam.nn_module.class_idx.cpu(), inferred)

        fea_shape = cam.feature_map_size(input_data["shape"], device=device, **kwargs)
        self.assertTupleEqual(fea_shape, input_data["feature_shape"])
        self.assertTupleEqual(result.shape, expected_shape)
        # check result is same whether class_idx=None is used or not
        result2 = cam(x=image, layer_idx=-1, class_idx=inferred, **kwargs)
        assert_allclose(result, result2)

    @parameterized.expand(TESTS_BATCH)
    def test_batch_matches_single_sample(self, cam_class):
        """
        The maps of a batch equal the maps computed one image at a time, for ``class_idx`` given as
        ``None`` (per-image argmax), an integer (same class for every image) and a tensor (one class per image).
        """
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model = DenseNet(
            spatial_dims=3, in_channels=1, out_channels=3, init_features=2, growth_rate=2, block_config=(6,)
        )
        model.to(device)
        model.eval()
        cam = cam_class(nn_module=model, target_layers="class_layers.relu")
        image = torch.rand((3, 1, 6, 6, 6), device=device)
        for class_idx in (None, 1, torch.tensor([0, 2, 1], device=device)):
            batch = cam(x=image, class_idx=class_idx)
            single = []
            for i in range(image.shape[0]):
                class_idx_i = int(class_idx[i]) if isinstance(class_idx, torch.Tensor) else class_idx
                single.append(cam(x=image[i : i + 1], class_idx=class_idx_i))
            assert_allclose(batch, torch.cat(single), rtol=1e-4, atol=1e-5)

    @parameterized.expand(TESTS_ILL)
    def test_ill(self, cam_class):
        model = DenseNet121(spatial_dims=2, in_channels=1, out_channels=3)
        for name, x in model.named_parameters():
            if "features" in name:
                x.requires_grad = False
        cam = cam_class(nn_module=model, target_layers="class_layers.relu")
        image = torch.rand((2, 1, 48, 64))
        with self.assertRaises(IndexError):
            cam(x=image)


if __name__ == "__main__":
    unittest.main()

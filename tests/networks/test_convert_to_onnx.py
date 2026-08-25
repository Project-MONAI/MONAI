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

import itertools
import platform
import unittest

import torch
from parameterized import parameterized

from monai.networks import convert_to_onnx
from monai.networks.nets import (
    UNETR,
    AttentionUnet,
    BasicUNet,
    BasicUNetPlusPlus,
    DenseNet,
    DynUNet,
    FullyConnectedNet,
    HighResNet,
    SegResNet,
    SEResNet50,
    UNet,
    VNet,
    resnet10,
)
from tests.test_utils import SkipIfNoModule, optional_import, skip_if_quick

onnx, _ = optional_import("onnx")

TORCH_DEVICE_OPTIONS = ["cpu"]

# FIXME: CUDA seems to produce different model outputs during testing vs. ONNX outputs, use CPU only for now
# if torch.cuda.is_available():
#     TORCH_DEVICE_OPTIONS.append("cuda")

TESTS = list(itertools.product(TORCH_DEVICE_OPTIONS, [True, False], [True, False]))
TESTS_ORT = list(itertools.product(TORCH_DEVICE_OPTIONS, [True]))
TESTS_TRACE = list(itertools.product(TORCH_DEVICE_OPTIONS, [True, False]))

ON_AARCH64 = platform.machine() == "aarch64"
if ON_AARCH64:
    rtol, atol = 1e-1, 1e-2
else:
    rtol, atol = 1e-2, 1e-2


def _check_ort_available(test_case):
    """Skip the test if onnxruntime is not installed.

    Args:
        test_case: the ``unittest.TestCase`` instance to call ``skipTest`` on
            when onnxruntime is unavailable.

    Raises:
        unittest.SkipTest: when onnxruntime cannot be imported.
    """
    _, has_onnxruntime = optional_import("onnxruntime")
    if not has_onnxruntime:
        test_case.skipTest("onnxruntime is not installed probably due to python version >= 3.11.")


@SkipIfNoModule("onnx")
@skip_if_quick
class TestConvertToOnnx(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_unet(self, device, use_trace, use_ort):
        """Test converting UNet to ONNX."""
        if use_ort:
            _, has_onnxruntime = optional_import("onnxruntime")
            if not has_onnxruntime:
                self.skipTest("onnxruntime is not installed probably due to python version >= 3.11.")
        model = UNet(
            spatial_dims=2, in_channels=1, out_channels=3, channels=(16, 32, 64), strides=(2, 2), num_res_units=0
        )

        onnx_model = convert_to_onnx(
            model=model,
            inputs=[torch.randn((16, 1, 32, 32), requires_grad=False)],
            input_names=["x"],
            output_names=["y"],
            verify=True,
            device=device,
            use_ort=use_ort,
            use_trace=use_trace,
            rtol=rtol,
            atol=atol,
        )
        self.assertTrue(isinstance(onnx_model, onnx.ModelProto))

    @parameterized.expand(TESTS_ORT)
    def test_seg_res_net(self, device, use_ort):
        """Test converting SetResNet to ONNX."""
        if use_ort:
            _, has_onnxruntime = optional_import("onnxruntime")
            if not has_onnxruntime:
                self.skipTest("onnxruntime is not installed probably due to python version >= 3.11.")
        model = SegResNet(
            spatial_dims=3,
            init_filters=32,
            in_channels=1,
            out_channels=105,
            dropout_prob=0.2,
            act=("RELU", {"inplace": True}),
            norm=("GROUP", {"num_groups": 8}),
            norm_name="",
            num_groups=8,
            use_conv_final=True,
            blocks_down=[1, 2, 2, 4],
            blocks_up=[1, 1, 1],
        )
        onnx_model = convert_to_onnx(
            model=model,
            inputs=[torch.randn((1, 1, 24, 24, 24), requires_grad=False)],
            input_names=["x"],
            output_names=["y"],
            verify=True,
            device=device,
            use_ort=use_ort,
            use_trace=True,
            rtol=rtol,
            atol=atol,
        )
        self.assertTrue(isinstance(onnx_model, onnx.ModelProto))

    @parameterized.expand(TESTS_TRACE)
    def test_dynunet(self, device, use_ort):
        """Test converting DynUNet to ONNX with and without ORT verification.

        Args:
            device: torch device string (e.g. ``"cpu"``).
            use_ort: if ``True``, verify via onnxruntime; if ``False``, verify
                via ``onnx.reference.ReferenceEvaluator``. Skipped when
                onnxruntime is unavailable.
        """
        if use_ort:
            _check_ort_available(self)
        model = DynUNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            kernel_size=[3, 3, 3],
            strides=[1, 2, 2],
            upsample_kernel_size=[2, 2],
        )
        onnx_model = convert_to_onnx(
            model=model,
            inputs=[torch.randn((1, 1, 32, 32, 32), requires_grad=False)],
            input_names=["x"],
            output_names=["y"],
            verify=True,
            device=device,
            use_ort=use_ort,
            use_trace=True,
            rtol=rtol,
            atol=atol,
        )
        self.assertTrue(isinstance(onnx_model, onnx.ModelProto))

    @parameterized.expand(TESTS_TRACE)
    def test_attention_unet(self, device, use_ort):
        """Test converting AttentionUnet to ONNX with and without ORT verification.

        Args:
            device: torch device string (e.g. ``"cpu"``).
            use_ort: if ``True``, verify via onnxruntime; if ``False``, verify
                via ``onnx.reference.ReferenceEvaluator``. Skipped when
                onnxruntime is unavailable.
        """
        if use_ort:
            _check_ort_available(self)
        model = AttentionUnet(spatial_dims=3, in_channels=1, out_channels=2, channels=(16, 32, 64), strides=(2, 2))
        onnx_model = convert_to_onnx(
            model=model,
            inputs=[torch.randn((1, 1, 32, 32, 32), requires_grad=False)],
            input_names=["x"],
            output_names=["y"],
            verify=True,
            device=device,
            use_ort=use_ort,
            use_trace=True,
            rtol=rtol,
            atol=atol,
        )
        self.assertTrue(isinstance(onnx_model, onnx.ModelProto))

    @parameterized.expand(TESTS_TRACE)
    def test_basic_unet(self, device, use_ort):
        """Test converting BasicUNet to ONNX with and without ORT verification.

        Args:
            device: torch device string (e.g. ``"cpu"``).
            use_ort: if ``True``, verify via onnxruntime; if ``False``, verify
                via ``onnx.reference.ReferenceEvaluator``. Skipped when
                onnxruntime is unavailable.
        """
        if use_ort:
            _check_ort_available(self)
        model = BasicUNet(spatial_dims=3, in_channels=1, out_channels=2, features=(8, 8, 16, 32, 64, 8))
        onnx_model = convert_to_onnx(
            model=model,
            inputs=[torch.randn((1, 1, 32, 32, 32), requires_grad=False)],
            input_names=["x"],
            output_names=["y"],
            verify=True,
            device=device,
            use_ort=use_ort,
            use_trace=True,
            rtol=rtol,
            atol=atol,
        )
        self.assertTrue(isinstance(onnx_model, onnx.ModelProto))

    @parameterized.expand(TESTS_TRACE)
    def test_basic_unet_plus_plus(self, device, use_ort):
        """Test converting BasicUNetPlusPlus to ONNX with and without ORT verification.

        Args:
            device: torch device string (e.g. ``"cpu"``).
            use_ort: if ``True``, verify via onnxruntime; if ``False``, verify
                via ``onnx.reference.ReferenceEvaluator``. Skipped when
                onnxruntime is unavailable.
        """
        if use_ort:
            _check_ort_available(self)
        model = BasicUNetPlusPlus(
            spatial_dims=3, in_channels=1, out_channels=2, features=(8, 8, 16, 32, 64, 8), deep_supervision=False
        )
        onnx_model = convert_to_onnx(
            model=model,
            inputs=[torch.randn((1, 1, 32, 32, 32), requires_grad=False)],
            input_names=["x"],
            output_names=["y"],
            verify=True,
            device=device,
            use_ort=use_ort,
            use_trace=True,
            rtol=rtol,
            atol=atol,
        )
        self.assertTrue(isinstance(onnx_model, onnx.ModelProto))

    @parameterized.expand(TESTS_TRACE)
    def test_vnet(self, device, use_ort):
        """Test converting VNet to ONNX with and without ORT verification.

        Args:
            device: torch device string (e.g. ``"cpu"``).
            use_ort: if ``True``, verify via onnxruntime; if ``False``, verify
                via ``onnx.reference.ReferenceEvaluator``. Skipped when
                onnxruntime is unavailable.
        """
        if use_ort:
            _check_ort_available(self)
        model = VNet(spatial_dims=3, in_channels=1, out_channels=1)
        onnx_model = convert_to_onnx(
            model=model,
            inputs=[torch.randn((1, 1, 32, 32, 32), requires_grad=False)],
            input_names=["x"],
            output_names=["y"],
            verify=True,
            device=device,
            use_ort=use_ort,
            use_trace=True,
            rtol=rtol,
            atol=atol,
        )
        self.assertTrue(isinstance(onnx_model, onnx.ModelProto))

    @parameterized.expand(TESTS_TRACE)
    def test_highresnet(self, device, use_ort):
        """Test converting HighResNet to ONNX with and without ORT verification.

        Args:
            device: torch device string (e.g. ``"cpu"``).
            use_ort: if ``True``, verify via onnxruntime; if ``False``, verify
                via ``onnx.reference.ReferenceEvaluator``. Skipped when
                onnxruntime is unavailable.
        """
        if use_ort:
            _check_ort_available(self)
        model = HighResNet(spatial_dims=3, in_channels=1, out_channels=2)
        onnx_model = convert_to_onnx(
            model=model,
            inputs=[torch.randn((1, 1, 16, 16, 16), requires_grad=False)],
            input_names=["x"],
            output_names=["y"],
            verify=True,
            device=device,
            use_ort=use_ort,
            use_trace=True,
            rtol=rtol,
            atol=atol,
        )
        self.assertTrue(isinstance(onnx_model, onnx.ModelProto))

    @parameterized.expand(TESTS_TRACE)
    def test_densenet(self, device, use_ort):
        """Test converting DenseNet to ONNX with and without ORT verification.

        Args:
            device: torch device string (e.g. ``"cpu"``).
            use_ort: if ``True``, verify via onnxruntime; if ``False``, verify
                via ``onnx.reference.ReferenceEvaluator``. Skipped when
                onnxruntime is unavailable.
        """
        if use_ort:
            _check_ort_available(self)
        model = DenseNet(
            spatial_dims=3, in_channels=1, out_channels=2, init_features=16, growth_rate=8, block_config=(2, 2, 2, 2)
        )
        onnx_model = convert_to_onnx(
            model=model,
            inputs=[torch.randn((1, 1, 32, 32, 32), requires_grad=False)],
            input_names=["x"],
            output_names=["y"],
            verify=True,
            device=device,
            use_ort=use_ort,
            use_trace=True,
            rtol=rtol,
            atol=atol,
        )
        self.assertTrue(isinstance(onnx_model, onnx.ModelProto))

    @parameterized.expand(TESTS_TRACE)
    def test_resnet(self, device, use_ort):
        """Test converting ResNet to ONNX with and without ORT verification.

        Args:
            device: torch device string (e.g. ``"cpu"``).
            use_ort: if ``True``, verify via onnxruntime; if ``False``, verify
                via ``onnx.reference.ReferenceEvaluator``. Skipped when
                onnxruntime is unavailable.
        """
        if use_ort:
            _check_ort_available(self)
        model = resnet10(pretrained=False, spatial_dims=3, n_input_channels=1, num_classes=2)
        onnx_model = convert_to_onnx(
            model=model,
            inputs=[torch.randn((1, 1, 32, 32, 32), requires_grad=False)],
            input_names=["x"],
            output_names=["y"],
            verify=True,
            device=device,
            use_ort=use_ort,
            use_trace=True,
            rtol=rtol,
            atol=atol,
        )
        self.assertTrue(isinstance(onnx_model, onnx.ModelProto))

    @parameterized.expand(TESTS_TRACE)
    def test_seresnet(self, device, use_ort):
        """Test converting SEResNet50 to ONNX with and without ORT verification.

        Args:
            device: torch device string (e.g. ``"cpu"``).
            use_ort: if ``True``, verify via onnxruntime; if ``False``, verify
                via ``onnx.reference.ReferenceEvaluator``. Skipped when
                onnxruntime is unavailable.
        """
        if use_ort:
            _check_ort_available(self)
        model = SEResNet50(layers=(1, 1, 1, 1), spatial_dims=3, in_channels=1, num_classes=2)
        onnx_model = convert_to_onnx(
            model=model,
            inputs=[torch.randn((1, 1, 32, 32, 32), requires_grad=False)],
            input_names=["x"],
            output_names=["y"],
            verify=True,
            device=device,
            use_ort=use_ort,
            use_trace=True,
            rtol=rtol,
            atol=atol,
        )
        self.assertTrue(isinstance(onnx_model, onnx.ModelProto))

    @parameterized.expand(TESTS_TRACE)
    def test_unetr(self, device, use_ort):
        """Test converting UNETR to ONNX with and without ORT verification.

        Args:
            device: torch device string (e.g. ``"cpu"``).
            use_ort: if ``True``, verify via onnxruntime; if ``False``, verify
                via ``onnx.reference.ReferenceEvaluator``. Skipped when
                onnxruntime is unavailable.
        """
        if use_ort:
            _check_ort_available(self)
        model = UNETR(
            in_channels=1,
            out_channels=2,
            img_size=(32, 32, 32),
            feature_size=8,
            hidden_size=128,
            mlp_dim=256,
            num_heads=8,
            spatial_dims=3,
        )
        onnx_model = convert_to_onnx(
            model=model,
            inputs=[torch.randn((1, 1, 32, 32, 32), requires_grad=False)],
            input_names=["x"],
            output_names=["y"],
            verify=True,
            device=device,
            use_ort=use_ort,
            use_trace=True,
            rtol=rtol,
            atol=atol,
        )
        self.assertTrue(isinstance(onnx_model, onnx.ModelProto))

    @parameterized.expand(TESTS_TRACE)
    def test_fully_connected_net(self, device, use_ort):
        """Test converting FullyConnectedNet to ONNX with and without ORT verification.

        Args:
            device: torch device string (e.g. ``"cpu"``).
            use_ort: if ``True``, verify via onnxruntime; if ``False``, verify
                via ``onnx.reference.ReferenceEvaluator``. Skipped when
                onnxruntime is unavailable.
        """
        if use_ort:
            _check_ort_available(self)
        model = FullyConnectedNet(in_channels=10, out_channels=2, hidden_channels=[20, 10])
        onnx_model = convert_to_onnx(
            model=model,
            inputs=[torch.randn((4, 10), requires_grad=False)],
            input_names=["x"],
            output_names=["y"],
            verify=True,
            device=device,
            use_ort=use_ort,
            use_trace=True,
            rtol=rtol,
            atol=atol,
        )
        self.assertTrue(isinstance(onnx_model, onnx.ModelProto))


if __name__ == "__main__":
    unittest.main()

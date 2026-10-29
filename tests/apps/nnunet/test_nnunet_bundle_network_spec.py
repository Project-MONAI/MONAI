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

import json
import os
import tempfile
import unittest
from unittest import mock

from monai.apps.nnunet.nnunet_bundle import _check_nnunet_network_spec, get_network_from_nnunet_plans
from monai.utils import optional_import

_, has_nnunet = optional_import("nnunetv2")

UNET = "dynamic_network_architectures.architectures.unet.PlainConvUNet"
IMPORTED_KWARGS = ["conv_op", "norm_op", "dropout_op", "nonlin"]


def _arch_kwargs(**overrides) -> dict:
    kwargs = {
        "n_stages": 2,
        "features_per_stage": [4, 8],
        "conv_op": "torch.nn.modules.conv.Conv2d",
        "kernel_sizes": [[3, 3], [3, 3]],
        "strides": [[1, 1], [2, 2]],
        "n_conv_per_stage": [1, 1],
        "n_conv_per_stage_decoder": [1],
        "conv_bias": True,
        "norm_op": "torch.nn.modules.instancenorm.InstanceNorm2d",
        "norm_op_kwargs": {"eps": 1e-5, "affine": True},
        "dropout_op": None,
        "dropout_op_kwargs": None,
        "nonlin": "torch.nn.LeakyReLU",
        "nonlin_kwargs": {"inplace": True},
    }
    kwargs.update(overrides)
    return kwargs


class TestNetworkSpecCheck(unittest.TestCase):
    def test_default_spec_is_accepted(self):
        with mock.patch("monai.apps.nnunet.nnunet_bundle.locate", return_value=None):
            _check_nnunet_network_spec(UNET, _arch_kwargs(), IMPORTED_KWARGS)

    def test_torch_nn_modules_are_accepted(self):
        _check_nnunet_network_spec("torch.nn.Identity", _arch_kwargs(), IMPORTED_KWARGS)

    def test_class_outside_allowed_modules_is_rejected(self):
        for name in (
            "numpy.savez",
            "os.system",
            "builtins.eval",
            "torch.nnx.Thing",
            "dynamic_network_architecturesx.A",
        ):
            with self.subTest(name=name):
                with mock.patch("monai.apps.nnunet.nnunet_bundle.locate") as locate:
                    with self.assertRaisesRegex(ValueError, "not in"):
                        _check_nnunet_network_spec(name, _arch_kwargs(), IMPORTED_KWARGS)
                locate.assert_not_called()

    def test_imported_argument_outside_allowed_modules_is_rejected(self):
        for key, value in (("nonlin", "subprocess.run"), ("conv_op", "os.system"), ("norm_op", 3)):
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, repr(key)):
                    _check_nnunet_network_spec("torch.nn.Identity", _arch_kwargs(**{key: value}), IMPORTED_KWARGS)

    def test_allowed_name_that_is_not_a_module_class_is_rejected(self):
        for name in ("torch.nn.functional.relu", "torch.nn.init.zeros_", "torch.nn.utils"):
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, "torch.nn.Module"):
                    _check_nnunet_network_spec(name, {}, [])

    def test_non_string_class_name_is_rejected(self):
        with self.assertRaises(ValueError):
            _check_nnunet_network_spec(None, {}, [])

    def test_custom_modules_can_be_allowed(self):
        with self.assertRaises(ValueError):
            _check_nnunet_network_spec("monai.networks.nets.BasicUNet", {}, [])
        _check_nnunet_network_spec("monai.networks.nets.BasicUNet", {}, [], allowed_modules=["monai.networks"])


def _plans(class_name: str = UNET, **kwarg_overrides) -> dict:
    resampling = "resample_data_or_seg_to_shape"
    return {
        "dataset_name": "Dataset001_Test",
        "plans_name": "nnUNetPlans",
        "original_median_spacing_after_transp": [1, 1],
        "original_median_shape_after_transp": [8, 8],
        "image_reader_writer": "SimpleITKIO",
        "transpose_forward": [0, 1],
        "transpose_backward": [0, 1],
        "experiment_planner_used": "ExperimentPlanner",
        "label_manager": "LabelManager",
        "foreground_intensity_properties_per_channel": {},
        "configurations": {
            "2d": {
                "data_identifier": "nnUNetPlans_2d",
                "preprocessor_name": "DefaultPreprocessor",
                "batch_size": 1,
                "patch_size": [8, 8],
                "median_image_size_in_voxels": [8, 8],
                "spacing": [1, 1],
                "normalization_schemes": ["ZScoreNormalization"],
                "use_mask_for_norm": [False],
                "resampling_fn_data": resampling,
                "resampling_fn_seg": resampling,
                "resampling_fn_probabilities": resampling,
                "resampling_fn_data_kwargs": {},
                "resampling_fn_seg_kwargs": {},
                "resampling_fn_probabilities_kwargs": {},
                "batch_dice": True,
                "architecture": {
                    "network_class_name": class_name,
                    "arch_kwargs": _arch_kwargs(**kwarg_overrides),
                    "_kw_requires_import": IMPORTED_KWARGS,
                },
            }
        },
    }


@unittest.skipUnless(has_nnunet, "Requires nnunetv2")
class TestGetNetworkFromPlans(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.plans_file = os.path.join(self._tmp.name, "plans.json")
        self.dataset_file = os.path.join(self._tmp.name, "dataset.json")
        with open(self.dataset_file, "w") as f:
            json.dump({"channel_names": {"0": "CT"}, "labels": {"background": 0, "a": 1}, "file_ending": ".nii.gz"}, f)

    def tearDown(self):
        self._tmp.cleanup()

    def _load(self, plans: dict):
        with open(self.plans_file, "w") as f:
            json.dump(plans, f)
        return get_network_from_nnunet_plans(self.plans_file, self.dataset_file, "2d")

    def test_default_plans_build_the_network(self):
        self.assertEqual(type(self._load(_plans())).__name__, "PlainConvUNet")

    def test_network_class_outside_allowed_modules_is_not_called(self):
        output = os.path.join(self._tmp.name, "out.npz")
        with self.assertRaisesRegex(ValueError, "not in"):
            self._load(_plans(class_name="numpy.savez", file=output))
        self.assertFalse(os.path.exists(output))

    def test_imported_argument_outside_allowed_modules_is_not_called(self):
        with mock.patch("os.system") as system:
            with self.assertRaisesRegex(ValueError, "'nonlin'"):
                self._load(_plans(nonlin="os.system"))
        system.assert_not_called()


if __name__ == "__main__":
    unittest.main()

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

import json
import os
import subprocess
import sys
import tempfile
import unittest

import nibabel as nib
import numpy as np
import yaml
from parameterized import parameterized

from monai.transforms import LoadImage

TEST_CASE_1 = [
    {
        "device": "$torch.device('cuda' if torch.cuda.is_available() else 'cpu')",
        "network_def": {
            "<name>": "UNet",
            "<args>": {
                "spatial_dims": 3,
                "in_channels": 1,
                "out_channels": 2,
                "channels": [16, 32, 64, 128, 256],
                "strides": [2, 2, 2, 2],
                "num_res_units": 2,
                "norm": "batch",
            },
        },
        "network": "will be overrided",
        "preprocessing": {
            "<name>": "Compose",
            "<args>": {
                "transforms": [
                    {"<name>": "LoadImaged", "<args>": {"keys": "image"}},
                    {"<name>": "EnsureChannelFirstd", "<args>": {"keys": "image"}},
                    {"<name>": "ScaleIntensityd", "<args>": {"keys": "image"}},
                    {"<name>": "EnsureTyped", "<args>": {"keys": "image"}},
                ]
            },
        },
        "dataset": {
            "<name>": "will be overrided",
            "<args>": {"data": "@<meta>#datalist", "transform": "@preprocessing"},  # test placeholger with `datalist`
        },
        "dataloader": {
            "<name>": "DataLoader",
            "<args>": {"dataset": "@dataset", "batch_size": 1, "shuffle": False, "num_workers": 4},
        },
        "inferer": {
            "<name>": "SlidingWindowInferer",
            "<args>": {"roi_size": [96, 96, 96], "sw_batch_size": 4, "overlap": 0.5},
        },
        "postprocessing": {
            "<name>": "Compose",
            "<args>": {
                "transforms": [
                    {"<name>": "Activationsd", "<args>": {"keys": "pred", "softmax": True}},
                    {"<name>": "AsDiscreted", "<args>": {"keys": "pred", "argmax": True}},
                    {
                        "<name>": "SaveImaged",
                        "<args>": {
                            "keys": "pred",
                            "meta_keys": "image_meta_dict",
                            "output_dir": "@<meta>#output_dir",  # test placeholger with `output_dir`
                        },
                    },
                ]
            },
        },
        "evaluator": {
            "<name>": "SupervisedEvaluator",
            "<args>": {
                "device": "@device",
                "val_data_loader": "@dataloader",
                "network": "@network",
                "inferer": "@inferer",
                "postprocessing": "@postprocessing",
                "amp": False,
            },
        },
    },
    (128, 128, 128),
]


class TestBundleRun(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1])
    def test_shape(self, config, expected_shape):
        test_image = np.random.rand(128, 128, 128)
        with tempfile.TemporaryDirectory() as tempdir:
            filename = os.path.join(tempdir, "image.nii")
            nib.save(nib.Nifti1Image(test_image, np.eye(4)), filename)

            # generate default args in a file
            def_args = os.path.join(tempdir, "def_args.json")
            with open(def_args, "w") as f:
                json.dump({"config_file": "will be overrided by `config_file` arg"}, f)

            meta = {"datalist": [{"image": filename}], "output_dir": tempdir, "window": (96, 96, 96)}
            # test YAML file
            meta_file = os.path.join(tempdir, "meta.yaml")
            with open(meta_file, "w") as f:
                yaml.safe_dump(meta, f)
            # test JSON file
            config_file = os.path.join(tempdir, "config.json")
            with open(config_file, "w") as f:
                json.dump(config, f)

            # test override with file, up case postfix
            overridefile1 = os.path.join(tempdir, "override1.JSON")
            with open(overridefile1, "w") as f:
                # test override with part of the overriding file
                json.dump({"move_net": "$@network_def.to(@device)"}, f)
            overridefile2 = os.path.join(tempdir, "override2.JSON")
            with open(overridefile2, "w") as f:
                # test override with the whole overriding file
                json.dump("Dataset", f)

            saver = LoadImage(image_only=True)

            # test with `monai.bundle.scripts` as CLI entry directly
            ret = subprocess.check_call(
                [
                    f"{sys.executable}",
                    "-m",
                    "monai.bundle.scripts",
                    "run",
                    "--meta_file",
                    f"{meta_file}",
                    "--config_file",
                    f"{config_file}",
                    # `fire` can not parse below string, will pass to `run` as a string representing a dict
                    "--override",
                    f"{{'postprocessing#<args>#transforms#2#<args>#output_postfix':'seg',\
                    'network':'<file>{overridefile1}#move_net','dataset#<name>':'<file>{overridefile2}'}}",
                    "--target",
                    "evaluator",
                ]
            )
            self.assertEqual(ret, 0)
            self.assertTupleEqual(saver(os.path.join(tempdir, "image", "image_seg.nii.gz")).shape, expected_shape)

            # here test the script with `google fire` tool as CLI
            ret = subprocess.check_call(
                [
                    f"{sys.executable}",
                    "-m",
                    "fire",
                    "monai.bundle",
                    "run",
                    "--meta_file",
                    f"{meta_file}",
                    "--config_file",
                    f"{config_file}",
                    "--override",
                    # `fire` can parse below string as a dictionary
                    f"{{'evaluator#<args>#amp':False,'network':'<file>{overridefile1}#move_net',\
                    'dataset#<name>':'<file>{overridefile2}'}}",
                    "--target",
                    "evaluator",
                ]
            )
            self.assertEqual(ret, 0)
            self.assertTupleEqual(saver(os.path.join(tempdir, "image", "image_trans.nii.gz")).shape, expected_shape)


if __name__ == "__main__":
    unittest.main()

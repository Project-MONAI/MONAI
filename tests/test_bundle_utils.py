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

import os
import sys
import shutil
import subprocess
import tempfile
import unittest

import torch

from monai.bundle.utils import load_bundle_config
from monai.networks.nets import UNet

metadata = """
{
    "test_value": 1,
    "test_list": [2,3]
}
"""

test_json = """
{
    "test_dict": {
        "a": 3,
        "b": "c"
    },
    "network_def": {
        "_target_": "UNet",
        "spatial_dims": 2,
        "in_channels": 1,
        "out_channels": 1,
        "channels": [4,8],
        "strides": [2]
    }
}
"""


class TestLoadBundleConfig(unittest.TestCase):
    def setUp(self):
        self.bundle_dir = tempfile.TemporaryDirectory()
        self.dir_name = self.bundle_dir.name + "/TestBundle"
        self.zip_file = self.bundle_dir.name + "/TestBundle.zip"
        self.ts_file = self.bundle_dir.name + "/TestBundle.ts"

        # create the directories for the bundle
        os.mkdir(self.dir_name)
        os.mkdir(self.dir_name + "/configs")
        os.mkdir(self.dir_name + "/models")

        # fill bundle configs

        with open(self.dir_name + "/configs/metadata.json", "w") as o:
            o.write(metadata)

        with open(self.dir_name + "/configs/test.json", "w") as o:
            o.write(test_json)

        # save network
        net = UNet(2, 1, 1, [4, 8], [2])
        torch.save(net.state_dict(), self.dir_name + "/models/model.pt")
        # torch.jit.script(net).save(self.dir_name+"/models/model.ts")

    def tearDown(self):
        self.bundle_dir.cleanup()

    def test_load_config_dir(self):
        p = load_bundle_config(self.dir_name, "test.json")

        self.assertEqual(p["_meta_"]["test_value"], 1)

        self.assertEqual(p["test_dict"]["b"], "c")

    def test_load_config_zip(self):
        # create a zip of the bundle
        shutil.make_archive(self.zip_file[:-4], "zip", self.bundle_dir.name)

        p = load_bundle_config(self.zip_file, "test.json")

        self.assertEqual(p["_meta_"]["test_value"], 1)

        self.assertEqual(p["test_dict"]["b"], "c")

    def test_load_config_ts(self):
        # create a Torchscript zip of the bundle
        cmd = ["python", "-m", "monai.bundle", "ckpt_export", "network_def", "--filepath", self.ts_file]
        cmd += ["--meta_file", self.dir_name + "/configs/metadata.json"]
        cmd += ["--config_file", self.dir_name + "/configs/test.json"]
        cmd += ["--ckpt_file", self.dir_name + "/models/model.pt"]
        
        try:
            subprocess.check_output(cmd,stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            print("Output:", e.output, file=sys.stderr)
            raise
            
        p = load_bundle_config(self.ts_file, "test.json")

        self.assertEqual(p["_meta_"]["test_value"], 1)

        self.assertEqual(p["test_dict"]["b"], "c")

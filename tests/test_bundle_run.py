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
from parameterized import parameterized

from monai.bundle import ConfigReader
from monai.transforms import LoadImage

TEST_CASE_1 = [os.path.join(os.path.dirname(__file__), "testing_data", "inference.json"), (128, 128, 128)]

TEST_CASE_2 = [os.path.join(os.path.dirname(__file__), "testing_data", "inference.yaml"), (128, 128, 128)]


class TestBundleRun(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1, TEST_CASE_2])
    def test_shape(self, config_file, expected_shape):
        test_image = np.random.rand(128, 128, 128)
        with tempfile.TemporaryDirectory() as tempdir:
            filename = os.path.join(tempdir, "image.nii")
            nib.save(nib.Nifti1Image(test_image, np.eye(4)), filename)

            reader = ConfigReader()
            # generate default args in a JSON file
            reader.read_config({"config_file": "will be overrided by `config_file` arg"})
            def_args = os.path.join(tempdir, "def_args.json")
            reader.export_config_file(filepath=def_args)

            meta = {"datalist": [{"image": filename}], "output_dir": tempdir, "window": (96, 96, 96)}
            # test YAML file
            reader.read_config(meta)
            meta_file = os.path.join(tempdir, "meta.yaml")
            reader.export_config_file(filepath=meta_file)

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

            if sys.platform == "win32":
                override = "'network':'$@network_def.to(@device)','dataset#<name>':'Dataset'"
            else:
                override = f"'network':'%{overridefile1}#move_net','dataset#<name>':'%{overridefile2}'"
            # test with `monai.bundle` as CLI entry directly
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
                    f"{{'postprocessing#<args>#transforms#2#<args>#output_postfix':'seg',{override}}}",
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
                    f"{{'evaluator#<args>#amp':False,{override}}}",
                    "--target",
                    "evaluator",
                ]
            )
            self.assertEqual(ret, 0)
            self.assertTupleEqual(saver(os.path.join(tempdir, "image", "image_trans.nii.gz")).shape, expected_shape)


if __name__ == "__main__":
    unittest.main()

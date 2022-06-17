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
import subprocess
import tempfile
import unittest

from parameterized import parameterized

from monai.bundle import ConfigParser
from tests.utils import skip_if_windows

TEST_CASE_1 = [
    os.path.join(os.path.dirname(__file__), "testing_data", "metadata.json"),
    os.path.join(os.path.dirname(__file__), "testing_data", "inference.json"),
]


@skip_if_windows
class TestVerifyNetwork(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1])
    def test_verify(self, meta_file, config_file):
        with tempfile.TemporaryDirectory() as tempdir:
            def_args = {"meta_file": "will be replaced by `meta_file` arg", "p": 2}
            def_args_file = os.path.join(tempdir, "def_args.json")
            ConfigParser.export_config_file(config=def_args, filepath=def_args_file)

            cmd = ["coverage", "run", "-m", "monai.bundle", "verify_net_in_out", "network_def", "--meta_file"]
            cmd += [meta_file, "--config_file", config_file, "-n", "4", "--any", "16", "--args_file", def_args_file]
            cmd += ["--_meta_#network_data_format#inputs#image#spatial_shape", "[16,'*','2**p*n']"]

            test_env = os.environ.copy()
            print(f"CUDA_VISIBLE_DEVICES in {__file__}", test_env.get("CUDA_VISIBLE_DEVICES"))
            subprocess.check_call(cmd, env=test_env)


if __name__ == "__main__":
    unittest.main()

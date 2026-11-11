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

import sys
import unittest
from io import StringIO
from unittest.mock import patch

from monai.config import print_debug_info
from monai.config.deviceconfig import get_gpu_info


class TestPrintInfo(unittest.TestCase):

    def test_print_info(self):
        out = StringIO()
        print_debug_info(file=out)
        self.assertGreater(out.tell(), 0)

    @patch("torch.backends.cudnn.version", lambda: 0)
    @patch("torch.cuda.device_count", lambda: 0)
    @patch("torch.cuda.is_available", lambda: True)
    @patch("torch.version.hip", "7.0.0")
    def test_hip_version(self):
        """A ROCm build reports the HIP version, where `torch.version.cuda` would be None."""
        output = get_gpu_info()
        self.assertEqual(output["HIP version"], "7.0.0")
        self.assertNotIn("CUDA version", output)

    @patch("torch.backends.cudnn.version", lambda: 0)
    @patch("torch.cuda.device_count", lambda: 0)
    @patch("torch.cuda.is_available", lambda: True)
    @patch("torch.version.hip", None)
    @patch("torch.version.cuda", "12.4")
    def test_cuda_version(self):
        """A CUDA build is unaffected and still reports the CUDA version."""
        output = get_gpu_info()
        self.assertEqual(output["CUDA version"], "12.4")
        self.assertNotIn("HIP version", output)

    @patch("torch.version.hip", "7.0.0")
    def test_check_env_hip_version(self):
        """check_env.check_torch_cuda() prints 'HIP version:' on a ROCm build."""
        import monai.config.check_env as ce

        buf = StringIO()
        with patch.object(sys, "stdout", buf):
            ce.check_torch_cuda()
        self.assertIn("HIP version:", buf.getvalue())
        self.assertNotIn("CUDA version:", buf.getvalue())


if __name__ == "__main__":
    unittest.main()

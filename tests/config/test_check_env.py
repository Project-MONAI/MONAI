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
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from unittest.mock import patch

from parameterized import parameterized

from monai.config import print_debug_info as _orig_print_debug_info
from monai.config.check_env import (
    check_monai,
    check_torch,
    check_torch_cuda,
    print_environment,
    print_environment_vars,
    print_platform,
)

ALL_FUNCS = [print_platform, print_environment_vars, print_environment, check_monai, check_torch, check_torch_cuda]


class TestPrintInfo(unittest.TestCase):

    @parameterized.expand(ALL_FUNCS)
    def test_func_output(self, func):
        out = StringIO()
        with redirect_stdout(out), redirect_stderr(out):
            # print_debug_info resolves argument `file=sys.stdout` at init time so won't be affected by redirect_stdout
            with patch("monai.config.print_debug_info", lambda: _orig_print_debug_info(out)):
                ret = func()

        self.assertGreater(out.tell(), 0)
        self.assertIsInstance(ret, (type(None), bool))


if __name__ == "__main__":
    unittest.main()

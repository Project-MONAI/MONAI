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
from unittest.mock import patch

from tests.test_utils import MODULE_PATH, SkipIfNoModule, get_matplotlib_baseline


@SkipIfNoModule("matplotlib")
class TestMatplotlibBaseline(unittest.TestCase):
    def test_renderer_boundary(self):
        cases = (
            ("3.9.4", False),
            ("3.10.9", False),
            ("3.10.99", False),
            ("3.11.0.dev0", True),
            ("3.11.0a1", True),
            ("3.11.0rc1", True),
            ("3.11.0", True),
            ("3.11.2", True),
            ("3.12.0.dev0", True),
            ("4.0.0", True),
        )
        for version, modern in cases:
            with self.subTest(version=version), patch("matplotlib.__version__", version):
                expected = MODULE_PATH / "tests" / "testing_data"
                if modern:
                    expected /= "matplotlib_3_11"
                self.assertEqual(get_matplotlib_baseline("matshow3d_test.png"), expected / "matshow3d_test.png")

    def test_missing_reference_does_not_fall_back(self):
        with patch("matplotlib.__version__", "3.11.2"):
            path = get_matplotlib_baseline("missing-reference.png")
        self.assertEqual(path.parent.name, "matplotlib_3_11")
        self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()

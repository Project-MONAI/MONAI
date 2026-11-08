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


class TestLazyOnnxImport(unittest.TestCase):
    """Regression test for #8455.

    ``onnx``/``onnx.reference``/``onnxruntime`` used to be imported at module
    scope in ``monai/networks/utils.py`` and ``monai/bundle/scripts.py`` via
    ``optional_import``, which imports eagerly. Because ``import monai``
    auto-loads ``monai.networks``, a broken or hanging onnx install (e.g.
    onnx 1.18 on Windows) would take down ``import monai`` with no error.

    The fix moves those imports inside the functions that use them, so neither
    module binds onnx at module scope any more. Assert that directly: it is
    deterministic and independent of which other optional packages happen to be
    installed (some of them import onnx transitively, so checking
    ``sys.modules`` after ``import monai`` is not a reliable signal).
    """

    def test_utils_does_not_bind_onnx_at_module_scope(self):
        import monai.networks.utils as utils

        for attr in ("onnx", "onnxreference", "onnxruntime"):
            self.assertFalse(
                hasattr(utils, attr),
                f"monai.networks.utils must not import {attr} at module scope (regression for #8455)",
            )

    def test_scripts_does_not_bind_onnx_at_module_scope(self):
        import monai.bundle.scripts as scripts

        self.assertFalse(
            hasattr(scripts, "onnx"), "monai.bundle.scripts must not import onnx at module scope (regression for #8455)"
        )


if __name__ == "__main__":
    unittest.main()

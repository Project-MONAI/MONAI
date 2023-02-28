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

from monai.handlers import ClearMLImageHandler
from monai.utils import optional_import

Task, has_clearml = optional_import("clearml", name="Task")


@unittest.skipUnless(has_clearml, "Requires 'clearml' installation")
class TestHandlerClearMLImageHandler(unittest.TestCase):
    def test_task_init(self):
        Task.set_offline(offline_mode=True)
        try:
            ClearMLImageHandler(
                project_name="MONAI",
                task_name="Default Task",
                output_uri=True,
                tags=None,
                reuse_last_task_id=True,
                continue_last_task=False,
                auto_connect_frameworks=True,
            )
        except Exception as exc:
            self.fail(exc)
        self.assertEqual(Task.current_task().name, "Default Task")
        self.assertEqual(Task.current_task()._project_name[1], "MONAI")


if __name__ == "__main__":
    unittest.main()

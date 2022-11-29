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

import sys
import unittest
import warnings


def _enter_pr_4800(self):
    """
    code from https://github.com/python/cpython/pull/4800
    """
    # The __warningregistry__'s need to be in a pristine state for tests
    # to work properly.
    for v in list(sys.modules.values()):
        if getattr(v, "__warningregistry__", None):
            v.__warningregistry__ = {}
    self.warnings_manager = warnings.catch_warnings(record=True)
    self.warnings = self.warnings_manager.__enter__()
    warnings.simplefilter("always", self.expected)
    return self


# FIXME: workaround for https://bugs.python.org/issue29620
try:
    # Suppression for issue #494:  tests/__init__.py:34: error: Cannot assign to a method
    unittest.case._AssertWarnsContext.__enter__ = _enter_pr_4800  # type: ignore
except AttributeError:
    pass

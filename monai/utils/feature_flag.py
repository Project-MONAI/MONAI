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

import os
from contextlib import contextmanager

FEATURE_FLAG_PREFIX = "MONAI_FEATURE_ENABLED_"


class FeatureFlag:
    def __init__(self, name: str, *, default: bool = False):
        self.name = name
        self._enabled: bool | None = None
        self.default = default

    def _get_from_env(self):
        return os.getenv(FEATURE_FLAG_PREFIX + self.name, None)

    @property
    def enabled(self) -> bool:
        if self._enabled is None:
            env = self._get_from_env()
            if env is None:
                self._enabled = self.default
            else:
                self._enabled = env.lower() in ["true", "1", "yes"]
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    def __str__(self):
        return f"{self.name}: {self.enabled}, default: {self.default}"


@contextmanager
def with_feature_flag(feature_flag: FeatureFlag, enabled: bool):  # type: ignore
    original = feature_flag.enabled
    feature_flag.enabled = enabled
    try:
        yield
    finally:
        feature_flag.enabled = original

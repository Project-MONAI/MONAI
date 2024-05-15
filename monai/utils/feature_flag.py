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


class FeatureFlag:
    """
    A feature flag used to use for enabling/disabling experimental features and other optional behaviors.
    A flag called {name} can be set by environment variable `MONAI_FEATURE_ENABLED_{name}`
    (values considered as `True` are "true", "1", "yes", case-insensitive) or
    by explicitly calling `enable()`/`disable()`.
    When first accessing its property `enabled` and no value has been explicitly set using `enable()`/`disable()`,
    it will set the value of the flag based on the environment variable and
    if that is not set uses the default value passed in the constructor.
    After this, the value can only be changed using `enable()`/`disable()`.

    Args:
        name: the name of the feature flag.
        default: the default value of the flag.
            Only used if the environment variable is not set and
            the flag has not been explicitly set using `enable()`/`disable()`.
            Kwarg only, default is `False`.
    """

    FEATURE_FLAG_PREFIX = "MONAI_FEATURE_ENABLED_"

    def __init__(self, name: str, *, default: bool = False):
        self.name = name
        self._enabled: bool | None = None
        self.default = default

    def _get_from_env(self):
        return os.getenv(FeatureFlag.FEATURE_FLAG_PREFIX + self.name, None)

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
    """
    A context manager to temporarily set the value of a feature flag.
    At the end of the context, the value of the flag will be restored to its original value.
    Useful for testing or for enabling/disabling a feature for a specific block of code.
    """
    original = feature_flag.enabled
    feature_flag.enabled = enabled
    try:
        yield
    finally:
        feature_flag.enabled = original

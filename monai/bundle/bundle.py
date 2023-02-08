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

from abc import ABC

from monai.apps.utils import get_logger

__all__ = ["Bundle", "ConfigBundle"]

logger = get_logger(module_name=__name__)


class Bundle(ABC):
    """
    Base class for the bundle structure.

    """

    def __init__(self, root: str) -> None:
        self.root = root

    def get_metadata(self):
        # TODO
        pass


class ConfigBundle(Bundle):
    def __init__(self, root: str) -> None:
        super().__init__(root)

    def verify_metadata(self):
        # TODO
        pass

    def verify_net_in_out(self):
        # TODO
        pass

    def ckpt_export(self):
        # TODO
        pass

    def init_bundle(self):
        # TODO
        pass

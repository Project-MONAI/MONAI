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
"""
Defines a generic factory class.
"""

from __future__ import annotations

from typing import Callable


class Factory:
    """
    Baseline factory object.
    """

    def __len__(self) -> int:
        """Returns the number of stored components."""
        return len(self.factories)

    def __contains__(self, name: str) -> bool:
        """Returns True if the given name is stored."""
        return name in self.factories

    @property
    def names(self) -> tuple[str, ...]:
        """
        Produces all factory names.
        """

        return tuple(self.factories)

    def add(self, *args) -> None:
        """
        Add a factory item.
        """
        raise NotImplementedError

    def factory_item(self, *args) -> Callable:
        """
        Decorator for adding a factory item with the given name and other associated information.
        """

        def _add(func: Callable) -> Callable:
            self.add(*args, func)
            return func

        return _add

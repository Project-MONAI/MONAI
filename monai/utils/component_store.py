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

from collections import namedtuple
from keyword import iskeyword
from textwrap import dedent, indent
from typing import Any, Iterable, TypeVar

from monai.utils.factory import Factory

T = TypeVar("T")


def is_variable(name):
    """Returns True if `name` is a valid Python variable name and also not a keyword."""
    return name.isidentifier() and not iskeyword(name)


class ComponentStore(Factory):
    """
    Represents a storage object for other objects (specifically functions) keyed to a name with a description.

    These objects act as global named places for storing components for objects parameterised by component names.
    Typically this is functions although other objects can be added. Printing a component store will produce a
    list of members along with their docstring information if present.

    Example:

    .. code-block:: python

        TestStore = ComponentStore("Test Store", "A test store for demo purposes")

        @TestStore.add_def("my_func_name", "Some description of your function")
        def _my_func(a, b):
            '''A description of your function here.'''
            return a * b

        print(TestStore)  # will print out name, description, and 'my_func_name' with the docstring

        func = TestStore["my_func_name"]
        result = func(7, 6)

    """

    _Component = namedtuple("Component", ("description", "value"))  # internal value pair

    def __init__(self, name: str, description: str) -> None:
        self.factories: dict[str, self._Component] = {}
        self.name: str = name
        self.description: str = description

        self.__doc__ = f"Component Store '{name}': {description}\n{self.__doc__ or ''}".strip()

    def add(self, name: str, desc: str, value: T) -> T:
        """Store the object `value` under the name `name` with description `desc`."""

        if not is_variable(name):
            raise ValueError("Name of component must be valid Python identifier")

        self.factories[name] = self._Component(desc, value)
        return value

    def __iter__(self) -> Iterable:
        """Yields name/component pairs."""
        for k, v in self.factories.items():
            yield k, v.value

    def __str__(self):
        result = f"Component Store '{self.name}': {self.description}\nAvailable components:"
        for k, v in self.factories.items():
            result += f"\n* {k}:"

            if hasattr(v.value, "__doc__"):
                doc = indent(dedent(v.value.__doc__.lstrip("\n").rstrip()), "    ")
                result += f"\n{doc}\n"
            else:
                result += f" {v.description}"

        return result

    def __getattr__(self, name: str) -> Any:
        """Returns the stored object under the given name."""
        if name in self.factories:
            return self.factories[name].value
        else:
            return self.__getattribute__(name)

    def __getitem__(self, name: str) -> Any:
        """Returns the stored object under the given name."""
        if name in self.factories:
            return self.factories[name].value
        else:
            raise ValueError(f"Component '{name}' not found")

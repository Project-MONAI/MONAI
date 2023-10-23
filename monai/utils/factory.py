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


class Factory:
    """
    Baseline factory object.
    """

    # def __init__(self) -> None:
    #     self.factories: dict[str, Callable] = {}
    #
    # @property
    # def names(self) -> tuple[str, ...]:
    #     """
    #     Produces all factory names.
    #     """
    #
    #     return tuple(self.factories)
    #
    # def add_factory_callable(self, name: str, func: Callable) -> None:
    #     """
    #     Add the factory function to this object under the given name.
    #     """
    #
    #     self.factories[name.upper()] = func
    #     self.__doc__ = (
    #         "The supported member"
    #         + ("s are: " if len(self.names) > 1 else " is: ")
    #         + ", ".join(f"``{name}``" for name in self.names)
    #         + ".\nPlease see :py:class:`monai.networks.layers.split_args` for additional args parsing."
    #     )
    #
    # def factory_function(self, name: str) -> Callable:
    #     """
    #     Decorator for adding a factory function with the given name.
    #     """
    #
    #     def _add(func: Callable) -> Callable:
    #         self.add_factory_callable(name, func)
    #         return func
    #
    #     return _add
    #
    # def get_constructor(self, factory_name: str, *args) -> Any:
    #     """
    #     Get the constructor for the given factory name and arguments.
    #
    #     Raises:
    #         TypeError: When ``factory_name`` is not a ``str``.
    #
    #     """
    #
    #     if not isinstance(factory_name, str):
    #         raise TypeError(f"factory_name must a str but is {type(factory_name).__name__}.")
    #
    #     func = look_up_option(factory_name.upper(), self.factories)
    #     return func(*args)
    #
    # def __getitem__(self, args) -> Any:
    #     """
    #     Get the given name or name/arguments pair. If `args` is a callable it is assumed to be the constructor
    #     itself and is returned, otherwise it should be the factory name or a pair containing the name and arguments.
    #     """
    #
    #     # `args[0]` is actually a type or constructor
    #     if callable(args):
    #         return args
    #
    #     # `args` is a factory name or a name with arguments
    #     if isinstance(args, str):
    #         name_obj, args = args, ()
    #     else:
    #         name_obj, *args = args
    #
    #     return self.get_constructor(name_obj, *args)
    #
    # def __getattr__(self, key):
    #     """
    #     If `key` is a factory name, return it, otherwise behave as inherited. This allows referring to factory names
    #     as if they were constants, eg. `Fact.FOO` for a factory Fact with factory function foo.
    #     """
    #
    #     if key in self.factories:
    #         return key
    #
    #     return super().__getattribute__(key)
    #
    #

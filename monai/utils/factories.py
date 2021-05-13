# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Callable, Dict, Tuple

__all__ = ["ObjectFactory"]


class ObjectFactory:
    """
    Factory object for creating objects, this uses given factory functions to actually produce the types or constructing
    callables. These functions are referred to by name and can be added at any time.


    Defines factories for creating objects in generic, extensible, and dimensionally independent ways. A separate factory
    object is created for each type of object, and factory functions keyed to names are added to these objects. Whenever
    an object is requested the factory name and any necessary arguments are passed to the factory object. The return value
    is typically a type but can be any callable producing an object.

    The factory objects contain functions keyed to names converted to upper case, these names can be referred to as members
    of the factory so that they can function as constant identifiers. eg. instance normalization is named `Norm.INSTANCE`.

    For example, to get a transpose convolution object the name is needed and then a dimension argument is provided which is
    passed to the factory function:

    .. code-block:: python

        dimension = 3
        name = Conv.CONVTRANS
        conv = Conv[name, dimension]

    This allows the `dimension` value to be set in the constructor, for example so that the dimensionality of a network is
    parameterizable. Not all factories require arguments after the name, the caller must be aware which are required.

    Defining new factories involves creating the object then associating it with factory functions:

    .. code-block:: python

        fact = ObjectFactory()

        @fact.factory_function('test')
        def make_something(x, y):
            # do something with x and y to choose which layer type to return
            return SomeLayerType
        ...

        # request object from factory TEST with 1 and 2 as values for x and y
        layer = fact[fact.TEST, 1, 2]

    Typically the caller of a factory would know what arguments to pass (ie. the dimensionality of the requested type) but
    can be parameterized with the factory name and the arguments to pass to the created type at instantiation time:

    .. code-block:: python

        def use_factory(fact_args):
            fact_name, type_args = split_args
            layer_type = fact[fact_name, 1, 2]
            return layer_type(**type_args)
        ...

        kw_args = {'arg0':0, 'arg1':True}
        layer = use_factory( (fact.TEST, kwargs) )
    """

    def __init__(self) -> None:
        self.factories: Dict[str, Callable] = {}

    @property
    def names(self) -> Tuple[str, ...]:
        """
        Produces all factory names.
        """

        return tuple(self.factories)

    def add_factory_callable(self, name: str, func: Callable) -> None:
        """
        Add the factory function to this object under the given name.
        """

        self.factories[name.upper()] = func
        self.__doc__ = (
            "The supported member"
            + ("s are: " if len(self.names) > 1 else " is: ")
            + ", ".join(f"``{name}``" for name in self.names)
            + ".\nPlease see :py:class:`monai.networks.layers.split_args` for additional args parsing."
        )

    def factory_function(self, name: str) -> Callable:
        """
        Decorator for adding a factory function with the given name.
        """

        def _add(func: Callable) -> Callable:
            self.add_factory_callable(name, func)
            return func

        return _add

    def get_constructor(self, factory_name: str, *args, **kwargs) -> Any:
        """
        Get the constructor for the given factory name and arguments.

        Raises:
            TypeError: When ``factory_name`` is not a ``str``.

        """

        if not isinstance(factory_name, str):
            raise TypeError(f"factory_name must a str but is {type(factory_name).__name__}.")

        fact = self.factories[factory_name.upper()]
        return fact(*args, **kwargs)

    def __getitem__(self, args) -> Any:
        """
        Get the given name or name/arguments pair. If `args` is a callable it is assumed to be the constructor
        itself and is returned, otherwise it should be the factory name or a pair containing the name and arguments.
        """

        # `args[0]` is actually a type or constructor
        if callable(args):
            return args

        # `args` is a factory name or a name with arguments
        if isinstance(args, str):
            name_obj, args = args, ()
        else:
            name_obj, *args = args

        return self.get_constructor(name_obj, *args)

    def __getattr__(self, key):
        """
        If `key` is a factory name, return it, otherwise behave as inherited. This allows referring to factory names
        as if they were constants, eg. `Fact.FOO` for a factory Fact with factory function foo.
        """

        if key in self.factories:
            return key

        return super().__getattribute__(key)

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

import inspect
import sys
import warnings
from abc import ABC, abstractmethod
from importlib import import_module
from typing import Any, Dict, List, Optional, Sequence, Union

from monai.apps.manifest.utils import is_expression, is_instantiable
from monai.utils import ensure_tuple, instantiate

__all__ = ["ComponentLocator", "ConfigItem", "ConfigComponent"]


class Instantiable(ABC):
    """
    Base class for instantiable object with module name and arguments.

    .. code-block:: python

        if not is_disabled():
            instantiate(module_name=resolve_module_name(), args=resolve_args())

    """

    @abstractmethod
    def resolve_module_name(self, *args: Any, **kwargs: Any):
        """
        Resolve the target module name, it should return an object class (or function) to be instantiated.
        """
        raise NotImplementedError(f"subclass {self.__class__.__name__} must implement this method.")

    @abstractmethod
    def resolve_args(self, *args: Any, **kwargs: Any):
        """
        Resolve the arguments, it should return arguments to be passed to the object when instantiating.
        """
        raise NotImplementedError(f"subclass {self.__class__.__name__} must implement this method.")

    @abstractmethod
    def is_disabled(self, *args: Any, **kwargs: Any) -> bool:
        """
        Return a boolean flag to indicate whether the object should be instantiated.
        """
        raise NotImplementedError(f"subclass {self.__class__.__name__} must implement this method.")

    @abstractmethod
    def instantiate(self, *args: Any, **kwargs: Any):
        """
        Instantiate the target component.
        """
        raise NotImplementedError(f"subclass {self.__class__.__name__} must implement this method.")


class ComponentLocator:
    """
    Scan all the available classes and functions in the MONAI package and map them with the module paths in a table.
    It's used to locate the module path for provided component name.

    Args:
        excludes: if any string of the `excludes` exists in the full module name, don't import this module.

    """

    MOD_START = "monai"

    def __init__(self, excludes: Optional[Union[Sequence[str], str]] = None):
        self.excludes = [] if excludes is None else ensure_tuple(excludes)
        self._components_table: Optional[Dict[str, List]] = None

    def _find_module_names(self) -> List[str]:
        """
        Find all the modules start with MOD_START and don't contain any of `excludes`.

        """
        return [
            m for m in sys.modules.keys() if m.startswith(self.MOD_START) and all(s not in m for s in self.excludes)
        ]

    def _find_classes_or_functions(self, modnames: Union[Sequence[str], str]) -> Dict[str, List]:
        """
        Find all the classes and functions in the modules with specified `modnames`.

        Args:
            modnames: names of the target modules to find all the classes and functions.

        """
        table: Dict[str, List] = {}
        # all the MONAI modules are already loaded by `load_submodules`
        for modname in ensure_tuple(modnames):
            try:
                # scan all the classes and functions in the module
                module = import_module(modname)
                for name, obj in inspect.getmembers(module):
                    if (inspect.isclass(obj) or inspect.isfunction(obj)) and obj.__module__ == modname:
                        if name not in table:
                            table[name] = []
                        table[name].append(modname)
            except ModuleNotFoundError:
                pass
        return table

    def get_component_module_name(self, name) -> Optional[Union[List[str], str]]:
        """
        Get the full module name of the class or function with specified ``name``.
        If target component name exists in multiple packages or modules, return a list of full module names.

        Args:
            name: name of the expected class or function.

        """
        if self._components_table is None:
            # init component and module mapping table
            self._components_table = self._find_classes_or_functions(self._find_module_names())

        mods: Optional[Union[List[str], str]] = self._components_table.get(name, None)
        if isinstance(mods, list) and len(mods) == 1:
            mods = mods[0]
        return mods


class ConfigItem:
    """
    Basic data structure to represent a configuration item.

    A `ConfigItem` instance can optionally have a string id, so that other items can refer to it.
    It has a build-in `config` property to store the configuration object.

    Args:
        config: content of a config item, can be objects of any types,
            a configuration resolver may interpret the content to generate a configuration object.
        id: optional ID name of the current config item, defaults to `None`.

    """

    def __init__(self, config: Any, id: Optional[str] = None) -> None:
        self.config = config
        self.id = id

    def get_id(self) -> Optional[str]:
        """
        Get the ID name of current config item, useful to identify config items during parsing.

        """
        return self.id

    def update_config(self, config: Any):
        """
        Replace the content of `self.config` with new `config`.
        A typical usage is to modify the initial config content at runtime.

        Args:
            config: content of a `ConfigItem`.

        """
        self.config = config

    def get_config(self):
        """
        Get the config content of current config item.

        """
        return self.config


class ConfigComponent(ConfigItem, Instantiable):
    """
    Subclass of :py:class:`monai.apps.ConfigItem`, this class uses a dictionary with string keys to
    represent a component of `class` or `function` and supports instantiation.

    Currently, four special keys (strings surrounded by ``<>``) are defined and interpreted beyond the regular literals:

        - class or function identifier of the python module, specified by one of the two keys.
            - ``"<name>"``: indicates build-in python classes or functions such as "LoadImageDict".
            - ``"<path>"``: full module name, such as "monai.transforms.LoadImageDict".
        - ``"<args>"``: input arguments to the python module.
        - ``"<disabled>"``: a boolean flag to indicate whether to skip the instantiation.

    .. code-block:: python

        locator = ComponentLocator(excludes=["modules_to_exclude"])
        config = {"<name>": "LoadImaged", "<args>": {"keys": ["image", "label"]}}

        configer = ConfigComponent(config, id="test", locator=locator)
        image_loader = configer.instantiate()
        print(image_loader)  # <monai.transforms.io.dictionary.LoadImaged object at 0x7fba7ad1ee50>

    Args:
        config: content of a config item.
        id: optional name of the current config item, defaults to `None`.
        locator: a `ComponentLocator` to convert a module name string into the actual python module.
            if `None`, a ``ComponentLocator(excludes=excludes)`` will be used.
        excludes: if ``locator`` is None, create a new ``ComponentLocator`` with ``excludes``.
            See also: :py:class:`monai.apps.manifest.ComponentLocator`.

    """

    def __init__(
        self,
        config: Any,
        id: Optional[str] = None,
        locator: Optional[ComponentLocator] = None,
        excludes: Optional[Union[Sequence[str], str]] = None,
    ) -> None:
        super().__init__(config=config, id=id)
        self.locator = ComponentLocator(excludes=excludes) if locator is None else locator

    def resolve_module_name(self):
        """
        Utility function used in `instantiate()` to resolve the target module name from current config content.
        The config content must have `<path>` or `<name>`.

        """
        config = dict(self.get_config())
        path = config.get("<path>")
        if path is not None:
            if "<name>" in config:
                warnings.warn(f"both '<path>' and '<name>', default to use '<path>': {path}.")
            return path

        name = config.get("<name>")
        if name is None:
            raise ValueError("must provide `<path>` or `<name>` of target component to instantiate.")

        module = self.locator.get_component_module_name(name)
        if module is None:
            raise ModuleNotFoundError(f"can not find component '{name}' in {self.locator.MOD_START} modules.")
        if isinstance(module, list):
            warnings.warn(
                f"there are more than 1 component have name `{name}`: {module}, use the first one `{module[0]}."
                f" if want to use others, please set its module path in `<path>` directly."
            )
            module = module[0]
        return f"{module}.{name}"

    def resolve_args(self):
        """
        Utility function used in `instantiate()` to resolve the arguments from current config content.

        """
        return self.get_config().get("<args>", {})

    def is_disabled(self):
        """
        Utility function used in `instantiate()` to check whether the current component is `disabled`.

        """
        return self.get_config().get("<disabled>", False)

    def instantiate(self, **kwargs) -> object:  # type: ignore
        """
        Instantiate component based on current config content.
        The target component must be a `class` or a `function`, otherwise, return `None`.

        Args:
            kwargs: args to override / add the config args when instantiation.

        """
        if not is_instantiable(self.get_config()) or self.is_disabled():
            # if not a class or function or marked as `disabled`, skip parsing and return `None`
            return None

        modname = self.resolve_module_name()
        args = self.resolve_args()
        args.update(kwargs)
        return instantiate(modname, **args)


class ConfigExpression(ConfigItem):
    """
    Subclass of :py:class:`monai.apps.ConfigItem`, the config item represents an executable expression
    string started with "$" mark, and support to execute based on python `eval()`, more details:
    https://docs.python.org/3/library/functions.html#eval.

    An example of config item: `{"test_fn": "$lambda x: x + 100"}}`

    The typical usage of the APIs:
    - Initialize / update config content.
    - `execute` the config content if it is expression.

    .. code-block:: python

        config = "$monai.data.list_data_collate"

        expression = ConfigExpression(config, id="test", globals={"monai": monai})
        dataloader = DataLoader(..., collate_fn=expression.execute())

    Args:
        config: content of a config item, can be a `dict`, `list`, `string`, `float`, `int`, etc.
        id: optional ID name of current config item, defaults to `None`.
        globals: to execute expression string, sometimes we need to provide the global variables which are
            referred in the expression string. for example: `globals={"monai": monai}` will be useful for
            config `{"collate_fn": "$monai.data.list_data_collate"}`.

    """

    def __init__(self, config: Any, id: Optional[str] = None, globals: Optional[Dict] = None) -> None:
        super().__init__(config=config, id=id)
        self.globals = globals

    def execute(self, locals: Optional[Dict] = None):
        """
        Excute current config content and return the result if it is expression, based on python `eval()`.
        For more details: https://docs.python.org/3/library/functions.html#eval.

        Args:
            locals: besides `globals`, may also have some local variables used in the expression at runtime.

        """
        value = self.get_config()
        if not is_expression(value):
            return None
        return eval(value[1:], self.globals, locals)

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
import os
import sys
import warnings
from abc import ABC, abstractmethod
from importlib import import_module
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

from monai.utils import ensure_tuple, instantiate

__all__ = ["ComponentLocator", "ConfigItem", "ConfigExpression", "ConfigComponent"]


class Instantiable(ABC):
    """
    Base class for an instantiable object.
    """

    @abstractmethod
    def is_disabled(self, *args: Any, **kwargs: Any) -> bool:
        """
        Return a boolean flag to indicate whether the object should be instantiated.
        """
        raise NotImplementedError(f"subclass {self.__class__.__name__} must implement this method.")

    @abstractmethod
    def instantiate(self, *args: Any, **kwargs: Any) -> object:
        """
        Instantiate the target component and return the instance.
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

    def get_component_module_name(self, name: str) -> Optional[Union[List[str], str]]:
        """
        Get the full module name of the class or function with specified ``name``.
        If target component name exists in multiple packages or modules, return a list of full module names.

        Args:
            name: name of the expected class or function.

        """
        if not isinstance(name, str):
            raise ValueError(f"`name` must be a valid string, but got: {name}.")
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
        id: name of the current config item, defaults to empty string.

    """

    def __init__(self, config: Any, id: str = "") -> None:
        self.config = config
        self.id = id

    def get_id(self) -> str:
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

    def __repr__(self) -> str:
        return str(self.config)


class ConfigComponent(ConfigItem, Instantiable):
    """
    Subclass of :py:class:`monai.apps.ConfigItem`, this class uses a dictionary with string keys to
    represent a component of `class` or `function` and supports instantiation.

    Currently, four special keys (strings surrounded by ``<>``) are defined and interpreted beyond the regular literals:

        - class or function identifier of the python module, specified by one of the two keys.
            - ``"<name>"``: indicates build-in python classes or functions such as "LoadImageDict".
            - ``"<path>"``: full module name, such as "monai.transforms.LoadImageDict".
        - ``"<args>"``: input arguments to the python module.
        - ``"<disabled>"``: a flag to indicate whether to skip the instantiation.

    .. code-block:: python

        locator = ComponentLocator(excludes=["modules_to_exclude"])
        config = {
            "<name>": "LoadImaged",
            "<args>": {
                "keys": ["image", "label"]
            }
        }

        configer = ConfigComponent(config, id="test", locator=locator)
        image_loader = configer.instantiate()
        print(image_loader)  # <monai.transforms.io.dictionary.LoadImaged object at 0x7fba7ad1ee50>

    Args:
        config: content of a config item.
        id: name of the current config item, defaults to empty string.
        locator: a ``ComponentLocator`` to convert a module name string into the actual python module.
            if `None`, a ``ComponentLocator(excludes=excludes)`` will be used.
        excludes: if ``locator`` is None, create a new ``ComponentLocator`` with ``excludes``.
            See also: :py:class:`monai.bundle.ComponentLocator`.

    """

    def __init__(
        self,
        config: Any,
        id: str = "",
        locator: Optional[ComponentLocator] = None,
        excludes: Optional[Union[Sequence[str], str]] = None,
    ) -> None:
        super().__init__(config=config, id=id)
        self.locator = ComponentLocator(excludes=excludes) if locator is None else locator

    @staticmethod
    def is_instantiable(config: Any) -> bool:
        """
        Check whether this config represents a `class` or `function` that is to be instantiated.

        Args:
            config: input config content to check.

        """
        return isinstance(config, Mapping) and ("<path>" in config or "<name>" in config)

    def resolve_module_name(self):
        """
        Resolve the target module name from current config content.
        The config content must have ``"<path>"`` or ``"<name>"``.
        When both are specified, ``"<path>"`` will be used.

        """
        config = dict(self.get_config())
        path = config.get("<path>")
        if path is not None:
            if not isinstance(path, str):
                raise ValueError(f"'<path>' must be a string, but got: {path}.")
            if "<name>" in config:
                warnings.warn(f"both '<path>' and '<name>', default to use '<path>': {path}.")
            return path

        name = config.get("<name>")
        if not isinstance(name, str):
            raise ValueError("must provide a string for `<path>` or `<name>` of target component to instantiate.")

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

    def is_disabled(self) -> bool:  # type: ignore
        """
        Utility function used in `instantiate()` to check whether to skip the instantiation.

        """
        _is_disabled = self.get_config().get("<disabled>", False)
        return _is_disabled.lower().strip() == "true" if isinstance(_is_disabled, str) else bool(_is_disabled)

    def instantiate(self, **kwargs) -> object:  # type: ignore
        """
        Instantiate component based on ``self.config`` content.
        The target component must be a `class` or a `function`, otherwise, return `None`.

        Args:
            kwargs: args to override / add the config args when instantiation.

        """
        if not self.is_instantiable(self.get_config()) or self.is_disabled():
            # if not a class or function or marked as `disabled`, skip parsing and return `None`
            return None

        modname = self.resolve_module_name()
        args = self.resolve_args()
        args.update(kwargs)
        return instantiate(modname, **args)


class ConfigExpression(ConfigItem):
    """
    Subclass of :py:class:`monai.apps.ConfigItem`, the `ConfigItem` represents an executable expression
    (execute based on ``eval()``).

    See also:

        - https://docs.python.org/3/library/functions.html#eval.

    For example:

    .. code-block:: python

        import monai
        from monai.bundle import ConfigExpression

        config = "$monai.__version__"
        expression = ConfigExpression(config, id="test", globals={"monai": monai})
        print(expression.execute())

    Args:
        config: content of a config item.
        id: name of current config item, defaults to empty string.
        globals: additional global context to evaluate the string.

    """

    prefix = "$"
    run_eval = False if os.environ.get("MONAI_EVAL_EXPR", "1") == "0" else True

    def __init__(self, config: Any, id: str = "", globals: Optional[Dict] = None) -> None:
        super().__init__(config=config, id=id)
        self.globals = globals

    def evaluate(self, locals: Optional[Dict] = None):
        """
        Execute the current config content and return the result if it is expression, based on Python `eval()`.
        For more details: https://docs.python.org/3/library/functions.html#eval.

        Args:
            locals: besides ``globals``, may also have some local symbols used in the expression at runtime.

        """
        value = self.get_config()
        if not ConfigExpression.is_expression(value):
            return None
        if not self.run_eval:
            return f"{value[len(self.prefix) :]}"
        return eval(value[len(self.prefix) :], self.globals, locals)

    @classmethod
    def is_expression(cls, config: Union[Dict, List, str]) -> bool:
        """
        Check whether the config is an executable expression string.
        Currently, a string starts with ``"$"`` character is interpreted as an expression.

        Args:
            config: input config content to check.

        """
        return isinstance(config, str) and config.startswith(cls.prefix)

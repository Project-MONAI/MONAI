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

import ast
import inspect
import sys
import warnings
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from importlib import import_module
from pprint import pformat
from typing import Any

from monai.bundle.utils import EXPR_KEY
from monai.utils import CompInitMode, ensure_tuple, first, instantiate, optional_import, run_debug, run_eval

__all__ = ["ComponentLocator", "ConfigItem", "ConfigExpression", "ConfigComponent", "Instantiable"]


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

    def __init__(self, excludes: Sequence[str] | str | None = None):
        self.excludes = [] if excludes is None else ensure_tuple(excludes)
        self._components_table: dict[str, list] | None = None

    def _find_module_names(self) -> list[str]:
        """
        Find all the modules start with MOD_START and don't contain any of `excludes`.

        """
        return [m for m in sys.modules if m.startswith(self.MOD_START) and all(s not in m for s in self.excludes)]

    def _find_classes_or_functions(self, modnames: Sequence[str] | str) -> dict[str, list]:
        """
        Find all the classes and functions in the modules with specified `modnames`.

        Args:
            modnames: names of the target modules to find all the classes and functions.

        """
        table: dict[str, list] = {}
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

    def get_component_module_name(self, name: str) -> list[str] | str | None:
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

        mods: list[str] | str | None = self._components_table.get(name)
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

    def update_config(self, config: Any) -> None:
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
        return f"{type(self).__name__}: \n{pformat(self.config)}"


class ConfigComponent(ConfigItem, Instantiable):
    """
    Subclass of :py:class:`monai.bundle.ConfigItem`, this class uses a dictionary with string keys to
    represent a component of `class` or `function` and supports instantiation.

    Currently, three special keys (strings surrounded by ``_``) are defined and interpreted beyond the regular literals:

        - class or function identifier of the python module, specified by ``"_target_"``,
          indicating a monai built-in Python class or function such as ``"LoadImageDict"``,
          or a full module name, e.g. ``"monai.transforms.LoadImageDict"``, or a callable, e.g. ``"$@model.forward"``.
        - ``"_requires_"`` (optional): specifies reference IDs (string starts with ``"@"``) or ``ConfigExpression``
          of the dependencies for this ``ConfigComponent`` object. These dependencies will be
          evaluated/instantiated before this object is instantiated.  It is useful when the
          component doesn't explicitly depend on the other `ConfigItems` via its arguments,
          but requires the dependencies to be instantiated/evaluated beforehand.
        - ``"_disabled_"`` (optional): a flag to indicate whether to skip the instantiation.
        - ``"_desc_"`` (optional): free text descriptions of the component for code readability.
        - ``"_mode_"`` (optional): operating mode for invoking the callable ``component`` defined by ``"_target_"``:

            - ``"default"``: returns ``component(**kwargs)``
            - ``"callable"``: returns ``component`` or, if ``kwargs`` are provided, ``functools.partial(component, **kwargs)``
            - ``"debug"``: returns ``pdb.runcall(component, **kwargs)``

    Other fields in the config content are input arguments to the python module.

    .. code-block:: python

        from monai.bundle import ComponentLocator, ConfigComponent

        locator = ComponentLocator(excludes=["modules_to_exclude"])
        config = {
            "_target_": "LoadImaged",
            "keys": ["image", "label"]
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

    non_arg_keys = {"_target_", "_disabled_", "_requires_", "_desc_", "_mode_"}

    def __init__(
        self,
        config: Any,
        id: str = "",
        locator: ComponentLocator | None = None,
        excludes: Sequence[str] | str | None = None,
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
        return isinstance(config, Mapping) and "_target_" in config

    def resolve_module_name(self):
        """
        Resolve the target module name from current config content.
        The config content must have ``"_target_"`` key.

        """
        config = dict(self.get_config())
        target = config.get("_target_")
        if not isinstance(target, str):
            return target  # for feature discussed in project-monai/monai#5852

        module = self.locator.get_component_module_name(target)
        if module is None:
            # target is the full module name, no need to parse
            return target

        if isinstance(module, list):
            warnings.warn(
                f"there are more than 1 component have name `{target}`: {module}, use the first one `{module[0]}."
                f" if want to use others, please set its full module path in `_target_` directly."
            )
            module = module[0]
        return f"{module}.{target}"

    def resolve_args(self):
        """
        Utility function used in `instantiate()` to resolve the arguments from current config content.

        """
        return {k: v for k, v in self.get_config().items() if k not in self.non_arg_keys}

    def is_disabled(self) -> bool:
        """
        Utility function used in `instantiate()` to check whether to skip the instantiation.

        """
        _is_disabled = self.get_config().get("_disabled_", False)
        return _is_disabled.lower().strip() == "true" if isinstance(_is_disabled, str) else bool(_is_disabled)

    def instantiate(self, **kwargs: Any) -> object:
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
        mode = self.get_config().get("_mode_", CompInitMode.DEFAULT)
        args = self.resolve_args()
        args.update(kwargs)
        return instantiate(modname, mode, **args)


class ConfigExpression(ConfigItem):
    """
    Subclass of :py:class:`monai.bundle.ConfigItem`, the `ConfigItem` represents an executable expression
    (execute based on ``eval()``, or import the module to the `globals` if it's an import statement).

    See also:

        - https://docs.python.org/3/library/functions.html#eval.

    For example:

    .. code-block:: python

        import monai
        from monai.bundle import ConfigExpression

        config = "$monai.__version__"
        expression = ConfigExpression(config, id="test", globals={"monai": monai})
        print(expression.evaluate())

    Args:
        config: content of a config item.
        id: name of current config item, defaults to empty string.
        globals: additional global context to evaluate the string.

    """

    prefix = EXPR_KEY
    run_eval = run_eval

    def __init__(self, config: Any, id: str = "", globals: dict | None = None) -> None:
        super().__init__(config=config, id=id)
        self.globals = globals if globals is not None else {}

    def _parse_import_string(self, import_string: str) -> Any | None:
        """parse single import statement such as "from monai.transforms import Resize"""
        node = first(ast.iter_child_nodes(ast.parse(import_string)))
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            return None
        if len(node.names) < 1:
            return None
        if len(node.names) > 1:
            warnings.warn(f"ignoring multiple import alias '{import_string}'.")
        name, asname = f"{node.names[0].name}", node.names[0].asname
        asname = name if asname is None else f"{asname}"
        if isinstance(node, ast.ImportFrom):
            self.globals[asname], _ = optional_import(f"{node.module}", name=f"{name}")
            return self.globals[asname]
        if isinstance(node, ast.Import):
            self.globals[asname], _ = optional_import(f"{name}")
            return self.globals[asname]
        return None

    def evaluate(self, globals: dict | None = None, locals: dict | None = None) -> str | Any | None:
        """
        Execute the current config content and return the result if it is expression, based on Python `eval()`.
        For more details: https://docs.python.org/3/library/functions.html#eval.

        Args:
            globals: besides ``self.globals``, other global symbols used in the expression at runtime.
            locals: besides ``globals``, may also have some local symbols used in the expression at runtime.

        """
        value = self.get_config()
        if not ConfigExpression.is_expression(value):
            return None
        optional_module = self._parse_import_string(value[len(self.prefix) :])
        if optional_module is not None:
            return optional_module
        if not self.run_eval:
            return f"{value[len(self.prefix) :]}"
        globals_ = dict(self.globals)
        if globals is not None:
            for k, v in globals.items():
                if k in globals_:
                    warnings.warn(f"the new global variable `{k}` conflicts with `self.globals`, override it.")
                globals_[k] = v
        if not run_debug:
            try:
                return eval(value[len(self.prefix) :], globals_, locals)
            except Exception as e:
                raise RuntimeError(f"Failed to evaluate {self}") from e
        warnings.warn(
            f"\n\npdb: value={value}\n"
            f"See also Debugger commands documentation: https://docs.python.org/3/library/pdb.html\n"
        )
        import pdb

        pdb.run(value[len(self.prefix) :], globals_, locals)
        return None

    @classmethod
    def is_expression(cls, config: dict | list | str) -> bool:
        """
        Check whether the config is an executable expression string.
        Currently, a string starts with ``"$"`` character is interpreted as an expression.

        Args:
            config: input config content to check.

        """
        return isinstance(config, str) and config.startswith(cls.prefix)

    @classmethod
    def is_import_statement(cls, config: dict | list | str) -> bool:
        """
        Check whether the config is an import statement (a special case of expression).

        Args:
            config: input config content to check.
        """
        if not cls.is_expression(config):
            return False
        if "import" not in config:
            return False
        return isinstance(
            first(ast.iter_child_nodes(ast.parse(f"{config[len(cls.prefix) :]}"))), (ast.Import, ast.ImportFrom)
        )

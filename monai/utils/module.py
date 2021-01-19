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

import inspect
import sys
from importlib import import_module
from pkgutil import walk_packages
from re import match
from typing import Any, Callable, List, Sequence, Tuple, Union

import torch

from .misc import ensure_tuple

OPTIONAL_IMPORT_MSG_FMT = "{}"

__all__ = [
    "InvalidPyTorchVersionError",
    "OptionalImportError",
    "exact_version",
    "export",
    "min_version",
    "optional_import",
    "load_submodules",
    "get_full_type_name",
    "has_option",
    "get_package_version",
    "get_torch_version_tuple",
    "PT_BEFORE_1_7",
]


def export(modname):
    """
    Make the decorated object a member of the named module. This will also add the object under its aliases if it has
    a `__aliases__` member, thus this decorator should be before the `alias` decorator to pick up those names. Alias
    names which conflict with package names or existing members will be ignored.
    """

    def _inner(obj):
        mod = import_module(modname)
        if not hasattr(mod, obj.__name__):
            setattr(mod, obj.__name__, obj)

            # add the aliases for `obj` to the target module
            for alias in getattr(obj, "__aliases__", ()):
                if not hasattr(mod, alias):
                    setattr(mod, alias, obj)

        return obj

    return _inner


def load_submodules(basemod, load_all: bool = True, exclude_pattern: str = "(.*[tT]est.*)|(_.*)"):
    """
    Traverse the source of the module structure starting with module `basemod`, loading all packages plus all files if
    `load_all` is True, excluding anything whose name matches `exclude_pattern`.
    """
    submodules = []
    err_mod: List[str] = []
    for importer, name, is_pkg in walk_packages(
        basemod.__path__, prefix=basemod.__name__ + ".", onerror=err_mod.append
    ):
        if (is_pkg or load_all) and name not in sys.modules and match(exclude_pattern, name) is None:
            try:
                mod = import_module(name)
                importer.find_module(name).load_module(name)
                submodules.append(mod)
            except OptionalImportError:
                pass  # could not import the optional deps., they are ignored

    return submodules, err_mod


def get_full_type_name(typeobj):
    module = typeobj.__module__
    if module is None or module == str.__class__.__module__:
        return typeobj.__name__  # Avoid reporting __builtin__
    return module + "." + typeobj.__name__


def min_version(the_module, min_version_str: str = "") -> bool:
    """
    Convert version strings into tuples of int and compare them.

    Returns True if the module's version is greater or equal to the 'min_version'.
    When min_version_str is not provided, it always returns True.
    """
    if min_version_str:
        mod_version = tuple(int(x) for x in the_module.__version__.split(".")[:2])
        required = tuple(int(x) for x in min_version_str.split(".")[:2])
        return mod_version >= required
    return True  # always valid version


def exact_version(the_module, version_str: str = "") -> bool:
    """
    Returns True if the module's __version__ matches version_str
    """
    return bool(the_module.__version__ == version_str)


class InvalidPyTorchVersionError(Exception):
    """
    Raised when called function or method requires a more recent
    PyTorch version than that installed.
    """

    def __init__(self, required_version, name):
        message = f"{name} requires PyTorch version {required_version} or later"
        super().__init__(message)


class OptionalImportError(ImportError):
    """
    Could not import APIs from an optional dependency.
    """


def optional_import(
    module: str,
    version: str = "",
    version_checker: Callable[..., bool] = min_version,
    name: str = "",
    descriptor: str = OPTIONAL_IMPORT_MSG_FMT,
    version_args=None,
    allow_namespace_pkg: bool = False,
) -> Tuple[Any, bool]:
    """
    Imports an optional module specified by `module` string.
    Any importing related exceptions will be stored, and exceptions raise lazily
    when attempting to use the failed-to-import module.

    Args:
        module: name of the module to be imported.
        version: version string used by the version_checker.
        version_checker: a callable to check the module version, Defaults to monai.utils.min_version.
        name: a non-module attribute (such as method/class) to import from the imported module.
        descriptor: a format string for the final error message when using a not imported module.
        version_args: additional parameters to the version checker.
        allow_namespace_pkg: whether importing a namespace package is allowed. Defaults to False.

    Returns:
        The imported module and a boolean flag indicating whether the import is successful.

    Examples::

        >>> torch, flag = optional_import('torch', '1.1')
        >>> print(torch, flag)
        <module 'torch' from 'python/lib/python3.6/site-packages/torch/__init__.py'> True

        >>> the_module, flag = optional_import('unknown_module')
        >>> print(flag)
        False
        >>> the_module.method  # trying to access a module which is not imported
        OptionalImportError: import unknown_module (No module named 'unknown_module').

        >>> torch, flag = optional_import('torch', '42', exact_version)
        >>> torch.nn  # trying to access a module for which there isn't a proper version imported
        OptionalImportError: import torch (requires version '42' by 'exact_version').

        >>> conv, flag = optional_import('torch.nn.functional', '1.0', name='conv1d')
        >>> print(conv)
        <built-in method conv1d of type object at 0x11a49eac0>

        >>> conv, flag = optional_import('torch.nn.functional', '42', name='conv1d')
        >>> conv()  # trying to use a function from the not successfully imported module (due to unmatched version)
        OptionalImportError: from torch.nn.functional import conv1d (requires version '42' by 'min_version').
    """

    tb = None
    exception_str = ""
    if name:
        actual_cmd = f"from {module} import {name}"
    else:
        actual_cmd = f"import {module}"
    try:
        pkg = __import__(module)  # top level module
        the_module = import_module(module)
        if not allow_namespace_pkg:
            is_namespace = getattr(the_module, "__file__", None) is None and hasattr(the_module, "__path__")
            if is_namespace:
                raise AssertionError
        if name:  # user specified to load class/function/... from the module
            the_module = getattr(the_module, name)
    except Exception as import_exception:  # any exceptions during import
        tb = import_exception.__traceback__
        exception_str = f"{import_exception}"
    else:  # found the module
        if version_args and version_checker(pkg, f"{version}", version_args):
            return the_module, True
        if not version_args and version_checker(pkg, f"{version}"):
            return the_module, True

    # preparing lazy error message
    msg = descriptor.format(actual_cmd)
    if version and tb is None:  # a pure version issue
        msg += f" (requires '{module} {version}' by '{version_checker.__name__}')"
    if exception_str:
        msg += f" ({exception_str})"

    class _LazyRaise:
        def __init__(self, *_args, **_kwargs):
            _default_msg = (
                f"{msg}."
                + "\n\nFor details about installing the optional dependencies, please visit:"
                + "\n    https://docs.monai.io/en/latest/installation.html#installing-the-recommended-dependencies"
            )
            if tb is None:
                self._exception = OptionalImportError(_default_msg)
            else:
                self._exception = OptionalImportError(_default_msg).with_traceback(tb)

        def __getattr__(self, name):
            """
            Raises:
                OptionalImportError: When you call this method.
            """
            raise self._exception

        def __call__(self, *_args, **_kwargs):
            """
            Raises:
                OptionalImportError: When you call this method.
            """
            raise self._exception

    return _LazyRaise(), False


def has_option(obj, keywords: Union[str, Sequence[str]]) -> bool:
    """
    Return a boolean indicating whether the given callable `obj` has the `keywords` in its signature.
    """
    if not callable(obj):
        return False
    sig = inspect.signature(obj)
    return all(key in sig.parameters for key in ensure_tuple(keywords))


def get_package_version(dep_name, default="NOT INSTALLED or UNKNOWN VERSION."):
    """
    Try to load package and get version. If not found, return `default`.

    If the package was already loaded, leave it. If wasn't previously loaded, unload it.
    """
    dep_ver = default
    dep_already_loaded = dep_name not in sys.modules

    dep, has_dep = optional_import(dep_name)
    if has_dep:
        if hasattr(dep, "__version__"):
            dep_ver = dep.__version__
        # if not previously loaded, unload it
        if not dep_already_loaded:
            del dep
            del sys.modules[dep_name]
    return dep_ver


def get_torch_version_tuple():
    """
    Returns:
        tuple of ints represents the pytorch major/minor version.
    """
    return tuple((int(x) for x in torch.__version__.split(".")[:2]))


PT_BEFORE_1_7 = True
ver, has_ver = optional_import("pkg_resources", name="parse_version")
try:
    if has_ver:
        PT_BEFORE_1_7 = ver(torch.__version__) < ver("1.7")
    else:
        PT_BEFORE_1_7 = get_torch_version_tuple() < (1, 7)
except (AttributeError, TypeError):
    pass

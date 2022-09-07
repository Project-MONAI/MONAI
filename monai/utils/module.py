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

import enum
import os
import re
import sys
import warnings
from functools import partial, wraps
from importlib import import_module
from inspect import isclass, isfunction, ismethod
from pkgutil import walk_packages
from pydoc import locate
from re import match
from types import FunctionType
from typing import Any, Callable, Collection, Hashable, Iterable, List, Mapping, Tuple, Union, cast

import torch

OPTIONAL_IMPORT_MSG_FMT = "{}"

__all__ = [
    "InvalidPyTorchVersionError",
    "OptionalImportError",
    "exact_version",
    "export",
    "damerau_levenshtein_distance",
    "look_up_option",
    "min_version",
    "optional_import",
    "require_pkg",
    "load_submodules",
    "instantiate",
    "get_full_type_name",
    "get_package_version",
    "get_torch_version_tuple",
    "version_leq",
    "pytorch_after",
]


def look_up_option(opt_str, supported: Union[Collection, enum.EnumMeta], default="no_default", print_all_options=True):
    """
    Look up the option in the supported collection and return the matched item.
    Raise a value error possibly with a guess of the closest match.

    Args:
        opt_str: The option string or Enum to look up.
        supported: The collection of supported options, it can be list, tuple, set, dict, or Enum.
        default: If it is given, this method will return `default` when `opt_str` is not found,
            instead of raising a `ValueError`. Otherwise, it defaults to `"no_default"`,
            so that the method may raise a `ValueError`.
        print_all_options: whether to print all available options when `opt_str` is not found. Defaults to True

    Examples:

    .. code-block:: python

        from enum import Enum
        from monai.utils import look_up_option
        class Color(Enum):
            RED = "red"
            BLUE = "blue"
        look_up_option("red", Color)  # <Color.RED: 'red'>
        look_up_option(Color.RED, Color)  # <Color.RED: 'red'>
        look_up_option("read", Color)
        # ValueError: By 'read', did you mean 'red'?
        # 'read' is not a valid option.
        # Available options are {'blue', 'red'}.
        look_up_option("red", {"red", "blue"})  # "red"

    Adapted from https://github.com/NifTK/NiftyNet/blob/v0.6.0/niftynet/utilities/util_common.py#L249
    """
    if not isinstance(opt_str, Hashable):
        raise ValueError(f"Unrecognized option type: {type(opt_str)}:{opt_str}.")
    if isinstance(opt_str, str):
        opt_str = opt_str.strip()
    if isinstance(supported, enum.EnumMeta):
        if isinstance(opt_str, str) and opt_str in {item.value for item in cast(Iterable[enum.Enum], supported)}:
            # such as: "example" in MyEnum
            return supported(opt_str)
        if isinstance(opt_str, enum.Enum) and opt_str in supported:
            # such as: MyEnum.EXAMPLE in MyEnum
            return opt_str
    elif isinstance(supported, Mapping) and opt_str in supported:
        # such as: MyDict[key]
        return supported[opt_str]
    elif isinstance(supported, Collection) and opt_str in supported:
        return opt_str

    if default != "no_default":
        return default

    # find a close match
    set_to_check: set
    if isinstance(supported, enum.EnumMeta):
        set_to_check = {item.value for item in cast(Iterable[enum.Enum], supported)}
    else:
        set_to_check = set(supported) if supported is not None else set()
    if not set_to_check:
        raise ValueError(f"No options available: {supported}.")
    edit_dists = {}
    opt_str = f"{opt_str}"
    for key in set_to_check:
        edit_dist = damerau_levenshtein_distance(f"{key}", opt_str)
        if edit_dist <= 3:
            edit_dists[key] = edit_dist

    supported_msg = f"Available options are {set_to_check}.\n" if print_all_options else ""
    if edit_dists:
        guess_at_spelling = min(edit_dists, key=edit_dists.get)  # type: ignore
        raise ValueError(
            f"By '{opt_str}', did you mean '{guess_at_spelling}'?\n"
            + f"'{opt_str}' is not a valid value.\n"
            + supported_msg
        )
    raise ValueError(f"Unsupported option '{opt_str}', " + supported_msg)


def damerau_levenshtein_distance(s1: str, s2: str):
    """
    Calculates the Damerau–Levenshtein distance between two strings for spelling correction.
    https://en.wikipedia.org/wiki/Damerau–Levenshtein_distance
    """
    if s1 == s2:
        return 0
    string_1_length = len(s1)
    string_2_length = len(s2)
    if not s1:
        return string_2_length
    if not s2:
        return string_1_length
    d = {(i, -1): i + 1 for i in range(-1, string_1_length + 1)}
    for j in range(-1, string_2_length + 1):
        d[(-1, j)] = j + 1

    for i, s1i in enumerate(s1):
        for j, s2j in enumerate(s2):
            cost = 0 if s1i == s2j else 1
            d[(i, j)] = min(
                d[(i - 1, j)] + 1, d[(i, j - 1)] + 1, d[(i - 1, j - 1)] + cost  # deletion  # insertion  # substitution
            )
            if i and j and s1i == s2[j - 1] and s1[i - 1] == s2j:
                d[(i, j)] = min(d[(i, j)], d[i - 2, j - 2] + cost)  # transposition

    return d[string_1_length - 1, string_2_length - 1]


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
                importer.find_module(name).load_module(name)  # type: ignore
                submodules.append(mod)
            except OptionalImportError:
                pass  # could not import the optional deps., they are ignored
            except ImportError as e:
                raise ImportError(
                    "Multiple versions of MONAI may have been installed,\n"
                    "please uninstall existing packages (both monai and monai-weekly) and install a version again.\n"
                    "See also: https://docs.monai.io/en/stable/installation.html\n"
                ) from e

    return submodules, err_mod


def instantiate(path: str, **kwargs):
    """
    Create an object instance or partial function from a class or function represented by string.
    `kwargs` will be part of the input arguments to the class constructor or function.
    The target component must be a class or a function, if not, return the component directly.

    Args:
        path: full path of the target class or function component.
        kwargs: arguments to initialize the class instance or set default args
            for `partial` function.

    """

    component = locate(path)
    if component is None:
        raise ModuleNotFoundError(f"Cannot locate class or function path: '{path}'.")
    try:
        if kwargs.pop("_debug_", False):
            print(f"\n\ndebug: instantiating: {component}\n\n")
            import pdb; pdb.set_trace()
        if isclass(component):
            return component(**kwargs)
        # support regular function, static method and class method
        if isfunction(component) or (ismethod(component) and isclass(getattr(component, "__self__", None))):
            return partial(component, **kwargs)
    except Exception as e:
        raise RuntimeError(f"Failed to instantiate '{path}' with kwargs: {kwargs}") from e

    warnings.warn(f"Component to instantiate must represent a valid class or function, but got {path}.")
    return component


def get_full_type_name(typeobj):
    """
    Utility to get the full path name of a class or object type.

    """
    module = typeobj.__module__
    if module is None or module == str.__class__.__module__:
        return typeobj.__name__  # Avoid reporting __builtin__
    return module + "." + typeobj.__name__


def min_version(the_module, min_version_str: str = "", *_args) -> bool:
    """
    Convert version strings into tuples of int and compare them.

    Returns True if the module's version is greater or equal to the 'min_version'.
    When min_version_str is not provided, it always returns True.
    """
    if not min_version_str or not hasattr(the_module, "__version__"):
        return True  # always valid version

    mod_version = tuple(int(x) for x in the_module.__version__.split(".")[:2])
    required = tuple(int(x) for x in min_version_str.split(".")[:2])
    return mod_version >= required


def exact_version(the_module, version_str: str = "", *_args) -> bool:
    """
    Returns True if the module's __version__ matches version_str
    """
    if not hasattr(the_module, "__version__"):
        warnings.warn(f"{the_module} has no attribute __version__ in exact_version check.")
        return False
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


def require_pkg(
    pkg_name: str, version: str = "", version_checker: Callable[..., bool] = min_version, raise_error: bool = True
):
    """
    Decorator function to check the required package installation.

    Args:
        pkg_name: required package name, like: "itk", "nibabel", etc.
        version: required version string used by the version_checker.
        version_checker: a callable to check the module version, defaults to `monai.utils.min_version`.
        raise_error: if True, raise `OptionalImportError` error if the required package is not installed
            or the version doesn't match requirement, if False, print the error in a warning.

    """

    def _decorator(obj):
        is_func = isinstance(obj, FunctionType)
        call_obj = obj if is_func else obj.__init__
        _, has = optional_import(module=pkg_name, version=version, version_checker=version_checker)

        @wraps(call_obj)
        def _wrapper(*args, **kwargs):
            if not has:
                err_msg = f"required package `{pkg_name}` is not installed or the version doesn't match requirement."
                if raise_error:
                    raise OptionalImportError(err_msg)
                else:
                    warnings.warn(err_msg)

            return call_obj(*args, **kwargs)

        if is_func:
            return _wrapper
        obj.__init__ = _wrapper
        return obj

    return _decorator


def get_package_version(dep_name, default="NOT INSTALLED or UNKNOWN VERSION."):
    """
    Try to load package and get version. If not found, return `default`.
    """
    dep, has_dep = optional_import(dep_name)
    if has_dep and hasattr(dep, "__version__"):
        return dep.__version__
    return default


def get_torch_version_tuple():
    """
    Returns:
        tuple of ints represents the pytorch major/minor version.
    """
    return tuple(int(x) for x in torch.__version__.split(".")[:2])


def version_leq(lhs: str, rhs: str):
    """
    Returns True if version `lhs` is earlier or equal to `rhs`.

    Args:
        lhs: version name to compare with `rhs`, return True if earlier or equal to `rhs`.
        rhs: version name to compare with `lhs`, return True if later or equal to `lhs`.

    """

    lhs, rhs = str(lhs), str(rhs)
    pkging, has_ver = optional_import("pkg_resources", name="packaging")
    if has_ver:
        try:
            return pkging.version.Version(lhs) <= pkging.version.Version(rhs)
        except pkging.version.InvalidVersion:
            return True

    def _try_cast(val: str):
        val = val.strip()
        try:
            m = match("(\\d+)(.*)", val)
            if m is not None:
                val = m.groups()[0]
                return int(val)
            return val
        except ValueError:
            return val

    # remove git version suffixes if present
    lhs = lhs.split("+", 1)[0]
    rhs = rhs.split("+", 1)[0]

    # parse the version strings in this basic way without `packaging` package
    lhs_ = map(_try_cast, lhs.split("."))
    rhs_ = map(_try_cast, rhs.split("."))

    for l, r in zip(lhs_, rhs_):
        if l != r:
            if isinstance(l, int) and isinstance(r, int):
                return l < r
            return f"{l}" < f"{r}"

    return True


def pytorch_after(major, minor, patch=0, current_ver_string=None) -> bool:
    """
    Compute whether the current pytorch version is after or equal to the specified version.
    The current system pytorch version is determined by `torch.__version__` or
    via system environment variable `PYTORCH_VER`.

    Args:
        major: major version number to be compared with
        minor: minor version number to be compared with
        patch: patch version number to be compared with
        current_ver_string: if None, `torch.__version__` will be used.

    Returns:
        True if the current pytorch version is greater than or equal to the specified version.
    """

    try:
        if current_ver_string is None:
            _env_var = os.environ.get("PYTORCH_VER", "")
            current_ver_string = _env_var if _env_var else torch.__version__
        ver, has_ver = optional_import("pkg_resources", name="parse_version")
        if has_ver:
            return ver(".".join((f"{major}", f"{minor}", f"{patch}"))) <= ver(f"{current_ver_string}")  # type: ignore
        parts = f"{current_ver_string}".split("+", 1)[0].split(".", 3)
        while len(parts) < 3:
            parts += ["0"]
        c_major, c_minor, c_patch = parts[:3]
    except (AttributeError, ValueError, TypeError):
        c_major, c_minor = get_torch_version_tuple()
        c_patch = "0"
    c_mn = int(c_major), int(c_minor)
    mn = int(major), int(minor)
    if c_mn != mn:
        return c_mn > mn
    is_prerelease = ("a" in f"{c_patch}".lower()) or ("rc" in f"{c_patch}".lower())
    c_p = 0
    try:
        p_reg = re.search(r"\d+", f"{c_patch}")
        if p_reg:
            c_p = int(p_reg.group())
    except (AttributeError, TypeError, ValueError):
        is_prerelease = True
    patch = int(patch)
    if c_p != patch:
        return c_p > patch  # type: ignore
    if is_prerelease:
        return False
    return True

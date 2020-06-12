# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib
from importlib import import_module
from pkgutil import walk_packages
from re import match
from typing import Any, Callable, Tuple

OPTIONAL_IMPORT_MSG_FMT = "No module named '{0}'"


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
    `loadAll` is True, excluding anything whose name matches `excludePattern`.
    """
    submodules = []

    for importer, name, is_pkg in walk_packages(basemod.__path__):
        if (is_pkg or load_all) and match(exclude_pattern, name) is None:
            mod = import_module(basemod.__name__ + "." + name)  # why do I need to do this first?
            importer.find_module(name).load_module(name)
            submodules.append(mod)

    return submodules


@export("monai.utils")
def get_full_type_name(typeobj):
    module = typeobj.__module__
    if module is None or module == str.__class__.__module__:
        return typeobj.__name__  # Avoid reporting __builtin__
    else:
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


def optional_import(
    module: str,
    version: str = "",
    version_checker: Callable = min_version,
    name: str = "",
    descriptor: str = OPTIONAL_IMPORT_MSG_FMT,
) -> Tuple[Any, bool]:
    """
    Imports an optional module specified by `module` string.
    Any importing related exceptions will be stored, and exceptions raise lazily
    when attempting to use the failed-to-import module.

    Args:
        module: name of the module to be imported.
        version: version string used by the version_checker.
        version_checker: a callable to check the module version, Defaults to monai.utils.min_version.
        name: attribute (such as method/class) to return from the imported module.
        descriptor: a format string for the final error message when using a not imported module.

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
        AttributeError: Optional import: No module named 'unknown_module'.

        >>> torch, flag = optional_import('torch', '42', exact_version)
        >>> torch.nn  # trying to access a module for which there isn't a proper version imported
        AttributeError: Optional import: No module named 'torch' (requires version '42', by 'exact_version').

        >>> conv, flag = optional_import('torch.nn.functional', '1.0', name='conv1d')
        >>> print(conv)
        <built-in method conv1d of type object at 0x11a49eac0>

        >>> conv, flag = optional_import('torch.nn.functional', '42', name='conv1d')
        >>> conv()  # trying to use a function from the not sucessfully imported module (due to unmatched version)
        AttributeError: Optional import: No module named 'torch.nn.functional' (requires version '42', by 'min_version').
    """

    tb = None
    exception_str = ""
    try:
        pkg = __import__(module)  # top level module
        the_module = importlib.import_module(module)
        if name:
            the_module = getattr(the_module, name)
    except Exception as import_exception:  # any exceptions during import
        tb = import_exception.__traceback__
        exception_str = f"{import_exception}"
    else:  # found the module
        if version_checker(pkg, version):
            return the_module, True

    # preparing lazy error message
    if version and tb is None:
        descriptor += " (requires version '{1}' by '{2}')"
    if exception_str:
        descriptor += f" ({exception_str})"
    msg = descriptor.format(module, version, version_checker.__name__)

    class _LazyRaise:
        def __init__(self, msg: str, trace_back=None):
            self.msg = msg
            self.trace_back = trace_back

        def __getattr__(self, name):
            if self.trace_back is None:
                raise AttributeError(f"Optional import: {self.msg}.")
            raise AttributeError(f"Optional import: {self.msg}.").with_traceback(self.trace_back)

        def __call__(self, *_args, **_kwargs):
            if self.trace_back is None:
                raise AttributeError(f"Optional import: {self.msg}.")
            raise AttributeError(f"Optional import: {self.msg}.").with_traceback(self.trace_back)

    return _LazyRaise(msg, tb), False

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

"""
This module is written for configurable workflow, not currently in use.
"""

import importlib
import inspect
import sys
import threading

alias_lock = threading.RLock()
GlobalAliases = {}

__all__ = ["alias", "resolve_name"]


def alias(*names):
    """
    Stores the decorated function or class in the global aliases table under the given names and as the `__aliases__`
    member of the decorated object. This new member will contain all alias names declared for that object.
    """

    def _outer(obj):
        for n in names:
            with alias_lock:
                GlobalAliases[n] = obj

        # set the member list __aliases__ to contain the alias names defined by the decorator for `obj`
        obj.__aliases__ = getattr(obj, "__aliases__", ()) + tuple(names)

        return obj

    return _outer


def resolve_name(name):
    """
    Search for the declaration (function or class) with the given name. This will first search the list of aliases to
    see if it was declared with this aliased name, then search treating `name` as a fully qualified name, then search
    the loaded modules for one having a declaration with the given name. If no declaration is found, raise ValueError.

    Raises:
        ValueError: When the module is not found.
        ValueError: When the module does not have the specified member.
        ValueError: When multiple modules with the declaration name are found.
        ValueError: When no module with the specified member is found.

    """
    # attempt to resolve an alias
    with alias_lock:
        obj = GlobalAliases.get(name, None)

    if name in GlobalAliases and obj is None:
        raise AssertionError

    # attempt to resolve a qualified name
    if obj is None and "." in name:
        modname, declname = name.rsplit(".", 1)

        try:
            mod = importlib.import_module(modname)
            obj = getattr(mod, declname, None)
        except ModuleNotFoundError:
            raise ValueError(f"Module {modname!r} not found.")

        if obj is None:
            raise ValueError(f"Module {modname!r} does not have member {declname!r}.")

    # attempt to resolve a simple name
    if obj is None:
        # Get all modules having the declaration/import, need to check here that getattr returns something which doesn't
        # equate to False since in places __getattr__ returns 0 incorrectly:
        # https://github.com/tensorflow/tensorboard/blob/a22566561d2b4fea408755a951ac9eaf3a156f8e/tensorboard/compat/tensorflow_stub/pywrap_tensorflow.py#L35  # noqa: B950
        mods = [m for m in list(sys.modules.values()) if getattr(m, name, None)]

        if len(mods) > 0:  # found modules with this declaration or import
            if len(mods) > 1:  # found multiple modules, need to determine if ambiguous or just multiple imports
                foundmods = {inspect.getmodule(getattr(m, name)) for m in mods}  # resolve imports
                foundmods = {m for m in foundmods if m is not None}

                if len(foundmods) > 1:  # found multiple declarations with the same name
                    modnames = [m.__name__ for m in foundmods]
                    msg = f"Multiple modules ({modnames!r}) with declaration name {name!r} found, resolution is ambiguous."
                    raise ValueError(msg)
                mods = list(foundmods)

            obj = getattr(mods[0], name)

        if obj is None:
            raise ValueError(f"No module with member {name!r} found.")

    return obj

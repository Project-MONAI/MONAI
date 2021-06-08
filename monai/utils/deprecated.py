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
import warnings
from functools import wraps
from threading import Lock
from types import FunctionType
from typing import Optional

from .. import __version__

__all__ = ["deprecated", "deprecated_arg", "DeprecatedError"]

warned_set = set()
warned_lock = Lock()


class DeprecatedError(Exception):
    pass


def warn_deprecated(obj, msg):
    """
    Issue the warning message `msg` only once per process for the given object `obj`. When this function is called
    and `obj` is not in `warned_set`, it is added and the warning issued, if it's already present nothing happens.
    """
    if obj not in warned_set:  # ensure warning is issued only once per process
        warned_set.add(obj)
        warnings.warn(msg, category=DeprecationWarning, stacklevel=2)


def version_leq(lhs, rhs):
    """Returns True if version `lhs` and earlier or equal to `rhs`."""

    if "+" in lhs:
        lhs = lhs.split("+", 1)[0]

    if "+" in rhs:
        rhs = rhs.split("+", 1)[0]

    def _try_cast(val):
        val = val.strip()
        try:
            return int(val)
        except ValueError:
            return val

    # parse the version strings in this basic way to avoid needing the `packaging` package
    lhs = map(_try_cast, lhs.split("."))
    rhs = map(_try_cast, rhs.split("."))

    return all(l <= r for l, r in zip(lhs, rhs))


def deprecated(
    since: Optional[str] = None, removed: Optional[str] = None, msg_suffix: str = "", version_val=__version__
):
    """
    Marks a function or class as deprecated. If `since` is given this should be a version at or earlier than the
    current version and states at what version of the definition was marked as deprecated. If `removed` is given
    this can be any version and marks when the definition was removed. When the decorated definition is called,
    that is when the function is called or the class instantiated, a warning is issued if `since` is given and
    the current version is at or later than that given. An exception is instead raised if `removed` is given and
    the current version is at or later than that, or if neither `since` nor `removed` is provided.
    """

    is_deprecated = since is not None and version_leq(since, version_val)
    is_removed = removed is not None and version_leq(removed, version_val)

    def _decorator(obj):
        is_func = isinstance(obj, FunctionType)
        call_obj = obj if is_func else obj.__init__

        msg_prefix = f"{'Function' if is_func else 'Class'} `{obj.__name__}`"

        if is_removed:
            msg_infix = f"was removed in version {removed}."
        elif is_deprecated:
            msg_infix = f"has been deprecated since version {since}."
            if removed is not None:
                msg_infix += f" It will be removed in version {removed}."
        else:
            msg_infix = "has been deprecated."

        msg = f"{msg_prefix} {msg_infix} {msg_suffix}".strip()

        @wraps(call_obj)
        def _wrapper(*args, **kwargs):
            if is_deprecated:
                warn_deprecated(obj, msg)
            else:
                raise DeprecatedError(msg)

            return call_obj(*args, **kwargs)

        if is_func:
            return _wrapper
        else:
            obj.__init__ = _wrapper
            return obj

    return _decorator


def deprecated_arg(
    name, since: Optional[str] = None, removed: Optional[str] = None, msg_suffix: str = "", version_val=__version__
):
    """
    Marks a particular named argument of a callable as deprecated. The same conditions for `since` and `removed` as
    described in the `deprecated` decorator.
    """

    is_deprecated = since is not None and version_leq(since, version_val)
    is_removed = removed is not None and version_leq(removed, version_val)

    def _decorator(func):
        argname = f"{func.__name__}_{name}"

        msg_prefix = f"Argument `{name}`"

        if is_removed:
            msg_infix = f"was removed in version {removed}."
        elif is_deprecated:
            msg_infix = f"has been deprecated since version {since}."
            if removed is not None:
                msg_infix += f" It will be removed in version {removed}."
        else:
            msg_infix = "has been deprecated."

        msg = f"{msg_prefix} {msg_infix} {msg_suffix}".strip()

        sig = inspect.signature(func)

        @wraps(func)
        def _wrapper(*args, **kwargs):
            binding = sig.bind(*args, **kwargs).arguments

            positional_found = name in binding
            kw_found = "kwargs" in binding and name in binding["kwargs"]

            if positional_found or kw_found:
                if is_deprecated:
                    warn_deprecated(argname, msg)
                else:
                    raise DeprecatedError(msg)

            return func(*args, **kwargs)

        return _wrapper

    return _decorator

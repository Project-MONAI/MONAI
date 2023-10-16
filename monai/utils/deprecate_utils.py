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

import inspect
import sys
import warnings
from collections.abc import Callable
from functools import wraps
from types import FunctionType
from typing import Any, TypeVar

from monai.utils.module import version_leq

from .. import __version__

__all__ = ["deprecated", "deprecated_arg", "DeprecatedError", "deprecated_arg_default"]
T = TypeVar("T", type, Callable)


class DeprecatedError(Exception):
    pass


def warn_deprecated(obj, msg, warning_category=FutureWarning):
    """
    Issue the warning message `msg`.
    """
    warnings.warn(f"{obj}: {msg}", category=warning_category, stacklevel=2)


def deprecated(
    since: str | None = None,
    removed: str | None = None,
    msg_suffix: str = "",
    version_val: str = __version__,
    warning_category: type[FutureWarning] = FutureWarning,
) -> Callable[[T], T]:
    """
    Marks a function or class as deprecated. If `since` is given this should be a version at or earlier than the
    current version and states at what version of the definition was marked as deprecated. If `removed` is given
    this can be any version and marks when the definition was removed.

    When the decorated definition is called, that is when the function is called or the class instantiated,
    a `warning_category` is issued if `since` is given and the current version is at or later than that given.
    a `DeprecatedError` exception is instead raised if `removed` is given and the current version is at or later
    than that, or if neither `since` nor `removed` is provided.

    The relevant docstring of the deprecating function should also be updated accordingly,
    using the Sphinx directives such as `.. versionchanged:: version` and `.. deprecated:: version`.
    https://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html#directive-versionadded

    Args:
        since: version at which the definition was marked deprecated but not removed.
        removed: version at which the definition was/will be removed and no longer usable.
        msg_suffix: message appended to warning/exception detailing reasons for deprecation and what to use instead.
        version_val: (used for testing) version to compare since and removed against, default is MONAI version.
        warning_category: a warning category class, defaults to `FutureWarning`.

    Returns:
        Decorated definition which warns or raises exception when used
    """

    if since is not None and removed is not None and not version_leq(since, removed):
        raise ValueError(f"since must be less or equal to removed, got since={since}, removed={removed}.")
    is_not_yet_deprecated = since is not None and version_val != since and version_leq(version_val, since)
    if is_not_yet_deprecated:
        # smaller than `since`, do nothing
        return lambda obj: obj

    if since is None and removed is None:
        # raise a DeprecatedError directly
        is_removed = True
        is_deprecated = True
    else:
        # compare the numbers
        is_deprecated = since is not None and version_leq(since, version_val)
        is_removed = removed is not None and version_leq(removed, version_val)

    def _decorator(obj):
        is_func = isinstance(obj, FunctionType)
        call_obj = obj if is_func else obj.__init__

        msg_prefix = f"{'Function' if is_func else 'Class'} `{obj.__qualname__}`"

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
            if is_removed:
                raise DeprecatedError(msg)
            if is_deprecated:
                warn_deprecated(obj, msg, warning_category)

            return call_obj(*args, **kwargs)

        if is_func:
            return _wrapper
        obj.__init__ = _wrapper
        return obj

    return _decorator


def deprecated_arg(
    name: str,
    since: str | None = None,
    removed: str | None = None,
    msg_suffix: str = "",
    version_val: str = __version__,
    new_name: str | None = None,
    warning_category: type[FutureWarning] = FutureWarning,
) -> Callable[[T], T]:
    """
    Marks a particular named argument of a callable as deprecated. The same conditions for `since` and `removed` as
    described in the `deprecated` decorator.

    When the decorated definition is called, that is when the function is called or the class instantiated with args,
    a `warning_category` is issued if `since` is given and the current version is at or later than that given.
    a `DeprecatedError` exception is instead raised if `removed` is given and the current version is at or later
    than that, or if neither `since` nor `removed` is provided.

    The relevant docstring of the deprecating function should also be updated accordingly,
    using the Sphinx directives such as `.. versionchanged:: version` and `.. deprecated:: version`.
    https://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html#directive-versionadded


    Args:
        name: name of position or keyword argument to mark as deprecated.
        since: version at which the argument was marked deprecated but not removed.
        removed: version at which the argument was/will be removed and no longer usable.
        msg_suffix: message appended to warning/exception detailing reasons for deprecation and what to use instead.
        version_val: (used for testing) version to compare since and removed against, default is MONAI version.
        new_name: name of position or keyword argument to replace the deprecated argument.
            if it is specified and the signature of the decorated function has a `kwargs`, the value to the
            deprecated argument `name` will be removed.
        warning_category: a warning category class, defaults to `FutureWarning`.

    Returns:
        Decorated callable which warns or raises exception when deprecated argument used.
    """

    if version_val.startswith("0+") or not f"{version_val}".strip()[0].isdigit():
        # version unknown, set version_val to a large value (assuming the latest version)
        version_val = f"{sys.maxsize}"
    if since is not None and removed is not None and not version_leq(since, removed):
        raise ValueError(f"since must be less or equal to removed, got since={since}, removed={removed}.")
    is_not_yet_deprecated = since is not None and version_val != since and version_leq(version_val, since)
    if is_not_yet_deprecated:
        # smaller than `since`, do nothing
        return lambda obj: obj
    if since is None and removed is None:
        # raise a DeprecatedError directly
        is_removed = True
        is_deprecated = True
    else:
        # compare the numbers
        is_deprecated = since is not None and version_leq(since, version_val)
        is_removed = removed is not None and version_val != f"{sys.maxsize}" and version_leq(removed, version_val)

    def _decorator(func):
        argname = f"{func.__module__} {func.__qualname__}:{name}"

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
            if new_name is not None and name in kwargs and new_name not in kwargs:
                # replace the deprecated arg "name" with "new_name"
                # if name is specified and new_name is not specified
                kwargs[new_name] = kwargs[name]
                try:
                    sig.bind(*args, **kwargs).arguments
                except TypeError:
                    # multiple values for new_name using both args and kwargs
                    kwargs.pop(new_name, None)
            binding = sig.bind(*args, **kwargs).arguments
            positional_found = name in binding
            kw_found = False
            for k, param in sig.parameters.items():
                if param.kind == inspect.Parameter.VAR_KEYWORD and k in binding and name in binding[k]:
                    kw_found = True
                    # if the deprecated arg is found in the **kwargs, it should be removed
                    kwargs.pop(name, None)

            if positional_found or kw_found:
                if is_removed:
                    raise DeprecatedError(msg)
                if is_deprecated:
                    warn_deprecated(argname, msg, warning_category)

            return func(*args, **kwargs)

        return _wrapper

    return _decorator


def deprecated_arg_default(
    name: str,
    old_default: Any,
    new_default: Any,
    since: str | None = None,
    replaced: str | None = None,
    msg_suffix: str = "",
    version_val: str = __version__,
    warning_category: type[FutureWarning] = FutureWarning,
) -> Callable[[T], T]:
    """
    Marks a particular arguments default of a callable as deprecated. It is changed from `old_default` to `new_default`
    in version `changed`.

    When the decorated definition is called, a `warning_category` is issued if `since` is given,
    the default is not explicitly set by the caller and the current version is at or later than that given.
    Another warning with the same category is issued if `changed` is given and the current version is at or later.

    The relevant docstring of the deprecating function should also be updated accordingly,
    using the Sphinx directives such as `.. versionchanged:: version` and `.. deprecated:: version`.
    https://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html#directive-versionadded


    Args:
        name: name of position or keyword argument where the default is deprecated/changed.
        old_default: name of the old default. This is only for the warning message, it will not be validated.
        new_default: name of the new default.
            It is validated that this value is not present as the default before version `replaced`.
            This means, that you can also use this if the actual default value is `None` and set later in the function.
            You can also set this to any string representation, e.g. `"calculate_default_value()"`
            if the default is calculated from another function.
        since: version at which the argument default was marked deprecated but not replaced.
        replaced: version at which the argument default was/will be replaced.
        msg_suffix: message appended to warning/exception detailing reasons for deprecation.
        version_val: (used for testing) version to compare since and removed against, default is MONAI version.
        warning_category: a warning category class, defaults to `FutureWarning`.

    Returns:
        Decorated callable which warns when deprecated default argument is not explicitly specified.
    """

    if version_val.startswith("0+") or not f"{version_val}".strip()[0].isdigit():
        # version unknown, set version_val to a large value (assuming the latest version)
        version_val = f"{sys.maxsize}"
    if since is not None and replaced is not None and not version_leq(since, replaced):
        raise ValueError(f"since must be less or equal to replaced, got since={since}, replaced={replaced}.")
    is_not_yet_deprecated = since is not None and version_val != since and version_leq(version_val, since)
    if is_not_yet_deprecated:
        # smaller than `since`, do nothing
        return lambda obj: obj
    if since is None and replaced is None:
        # raise a DeprecatedError directly
        is_replaced = True
        is_deprecated = True
    else:
        # compare the numbers
        is_deprecated = since is not None and version_leq(since, version_val)
        is_replaced = replaced is not None and version_val != f"{sys.maxsize}" and version_leq(replaced, version_val)

    def _decorator(func):
        argname = f"{func.__module__} {func.__qualname__}:{name}"

        msg_prefix = f" Current default value of argument `{name}={old_default}`"

        if is_replaced:
            msg_infix = f"was changed in version {replaced} from `{name}={old_default}` to `{name}={new_default}`."
        elif is_deprecated:
            msg_infix = f"has been deprecated since version {since}."
            if replaced is not None:
                msg_infix += f" It will be changed to `{name}={new_default}` in version {replaced}."
        else:
            msg_infix = f"has been deprecated from `{name}={old_default}` to `{name}={new_default}`."

        msg = f"{msg_prefix} {msg_infix} {msg_suffix}".strip()

        sig = inspect.signature(func)
        if name not in sig.parameters:
            raise ValueError(f"Argument `{name}` not found in signature of {func.__qualname__}.")
        param = sig.parameters[name]
        if param.default is inspect.Parameter.empty:
            raise ValueError(f"Argument `{name}` has no default value.")

        if param.default == new_default and not is_replaced:
            raise ValueError(
                f"Argument `{name}` was replaced to the new default value `{new_default}` before the specified version {replaced}."
            )

        @wraps(func)
        def _wrapper(*args, **kwargs):
            if name not in sig.bind(*args, **kwargs).arguments and is_deprecated:
                # arg was not found so the default value is used
                warn_deprecated(argname, msg, warning_category)

            return func(*args, **kwargs)

        return _wrapper

    return _decorator

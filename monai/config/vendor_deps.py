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

"""
Accelerator-vendor variants of the dependency lists declared in ``pyproject.toml``.

Those lists name the packages published for NVIDIA hardware (``cucim-cu*``, ``cupy-cuda*``,
``nvidia-*``). Other vendors publish equivalents under different distribution names, so a wheel
built against, say, a ROCm PyTorch needs different metadata from the same source tree.

A vendor is a :class:`Vendor` subclass defined in a sibling ``vendor_<name>.py`` module, which adds
an instance to :data:`REGISTRY` with :func:`register` when it is imported. :func:`active_vendor`
imports the modules named in :data:`PLUGIN_MODULES` and returns the vendor this build targets, if
any; ``setup.py`` and ``print_dependencies.py`` then call its ``apply_to_*`` methods.

``MONAI_VENDOR`` overrides detection: a registered name forces that vendor, ``none`` turns the
mechanism off without importing any vendor module, and ``auto`` (the default) detects.

To see what a vendor does to the lists, on any machine::

    MONAI_VENDOR=rocm python monai/config/print_dependencies.py all

``setup.py`` imports these modules directly rather than through the ``monai`` package, so at import
time they must depend on nothing but the standard library.
"""

from __future__ import annotations

import importlib
import os
import re
import warnings
from collections.abc import Mapping

__all__ = ["PLUGIN_MODULES", "REGISTRY", "Vendor", "active_vendor", "canonical_name", "register", "split_requirement"]

# Vendors by name, added by `register` as each module named in PLUGIN_MODULES is imported.
REGISTRY: dict[str, Vendor] = {}
PLUGIN_MODULES = ("vendor_rocm",)

# name, optional "[extras]", then everything else (version specifiers, markers, URLs).
_REQUIREMENT = re.compile(r"\s*(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)\s*(?:\[(?P<extras>[^]]*)\])?(?P<rest>.*)", re.S)


def canonical_name(name: str) -> str:
    """PEP 503 normalised distribution name."""
    return re.sub(r"[-_.]+", "-", name).lower()


def split_requirement(requirement: str) -> tuple[str, list[str], str]:
    """Split ``requirement`` into its name, its extras and the remaining specifier/marker text."""
    match = _REQUIREMENT.match(requirement)
    if match is None:
        return "", [], requirement
    extras = [e.strip() for e in (match["extras"] or "").split(",") if e.strip()]
    return match["name"], extras, match["rest"]


class Vendor:
    """
    An accelerator vendor and the changes its PyTorch build needs made to the dependency lists.

    A subclass sets ``name``, the identifier used by ``MONAI_VENDOR``, and ``substitutions``, mapping
    a PEP 503 normalised distribution name to its replacement requirement or to None to drop it, and
    overrides :meth:`detect`. The other hooks are optional.
    """

    name: str = ""
    substitutions: Mapping[str, str | None] = {}

    def detect(self) -> bool:
        """Whether the PyTorch this build sees is this vendor's. Runs on every build, so keep it cheap."""
        return False

    def hardware_present(self) -> bool:
        """Whether this vendor's hardware is in the machine, used only to warn that :meth:`detect` did not match."""
        return False

    def extras_for(self, name: str) -> list[str]:
        """Extras to add to the required dependency ``name``, e.g. device selectors."""
        return []

    def extra_requirements(self) -> list[str]:
        """Requirements to append to the required dependencies, e.g. a runtime version pin."""
        return []

    def substitute(self, requirements: list[str]) -> list[str]:
        """Apply ``substitutions`` to ``requirements``, dropping None entries and duplicates."""
        out: list[str] = []
        for requirement in requirements:
            key = canonical_name(split_requirement(requirement)[0])
            if key in self.substitutions:
                replacement = self.substitutions[key]
                if replacement is None:
                    continue
                requirement = replacement
            if requirement not in out:
                out.append(requirement)
        return out

    def apply_to_dependencies(self, dependencies: list[str]) -> list[str]:
        """Rewrite the required dependencies: substitute, add :meth:`extras_for`, append :meth:`extra_requirements`."""
        out: list[str] = []
        for requirement in self.substitute(dependencies):
            name, extras, rest = split_requirement(requirement)
            added = self.extras_for(canonical_name(name))
            if added:
                requirement = f"{name}[{','.join(sorted(set(extras) | set(added)))}]{rest}"
            out.append(requirement)
        return out + [requirement for requirement in self.extra_requirements() if requirement not in out]

    def apply_to_optional_dependencies(self, optional_dependencies: Mapping[str, list[str]]) -> dict[str, list[str]]:
        """Rewrite each optional-dependency group with :meth:`substitute`."""
        return {name: self.substitute(list(group)) for name, group in optional_dependencies.items()}


def register(vendor: Vendor) -> Vendor:
    """
    Add ``vendor`` to :data:`REGISTRY`; each plugin module calls this as it is imported.

    The substitution table is checked here because a mistake in it does not raise where it is used.
    ``key in substitutions`` is a substring test on a string, and lookups are canonicalised, so a table
    that is not a mapping, or a key that is not a PEP 503 name, would match nothing and leave another
    vendor's packages in this vendor's wheel.
    """
    if not vendor.name:
        raise ValueError(f"{type(vendor).__name__} must set a name")
    if not isinstance(vendor.substitutions, Mapping):
        raise TypeError(f"{vendor.name} substitutions must be a mapping")
    for key in vendor.substitutions:
        if not isinstance(key, str) or key != canonical_name(key):
            raise ValueError(
                f"{vendor.name} substitutions key {key!r} is not a PEP 503 normalised name, "
                f"so it would never match a requirement"
            )
    REGISTRY[vendor.name] = vendor
    return vendor


def active_vendor() -> Vendor | None:
    """The vendor whose packages this build needs, or None to use ``pyproject.toml`` as written."""
    requested = os.environ.get("MONAI_VENDOR", "").strip().lower() or "auto"
    if requested in ("none", "cpu"):
        return None

    for module in PLUGIN_MODULES:  # each registers its vendor as it is imported
        importlib.import_module(f".{module}" if __package__ else module, __package__ or None)

    if requested != "auto":
        if requested not in REGISTRY:
            raise ValueError(f"unknown MONAI_VENDOR {requested!r}; expected one of {sorted(REGISTRY)} or 'none'")
        return REGISTRY[requested]

    for vendor in REGISTRY.values():
        if vendor.detect():
            return vendor

    # pip's isolated build environment resolves torch afresh, and a plain PyPI wheel outranks a vendor
    # build of the same release, so a source install on vendor hardware can land here unintentionally.
    for vendor in REGISTRY.values():
        if vendor.hardware_present():
            warnings.warn(
                f"{vendor.name} hardware is present, but the torch this build sees is not a {vendor.name} build, "
                f"so {vendor.name} dependencies were not applied. Build against a {vendor.name} torch (for a "
                f"source install, `pip install --no-build-isolation` in an environment that already has one), "
                f"or set MONAI_VENDOR={vendor.name} to apply them regardless.",
                stacklevel=2,
            )
            break
    return None

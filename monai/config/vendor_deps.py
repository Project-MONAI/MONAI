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
Accelerator-vendor rewriting of the dependency lists declared in ``pyproject.toml``.

Those lists name the packages published for NVIDIA hardware (``cucim-cu*``, ``cupy-cuda*``,
``nvidia-*``). Other vendors publish equivalents under different distribution names, so a wheel
built against, say, a ROCm PyTorch needs different metadata from the same source tree.

This module owns the generic machinery -- requirement parsing, substitution, de-duplication -- and a
registry of vendor plugins. It contains no vendor-specific knowledge beyond a one-line detection
probe per entry in :data:`REGISTRY`.

A vendor plugin is a sibling module exposing:

``NAME``
    Identifier used by the ``MONAI_VENDOR`` environment variable.
``SUBSTITUTIONS``
    Mapping of canonical distribution name to its replacement requirement, or to None to drop it.
``extras_for(name)``
    Extras to add to the requirement for the distribution ``name``, e.g. device selectors.
``extra_requirements()``
    Requirements to append to the required dependencies, e.g. a runtime version pin.

Plugin modules are imported only once their registry probe matches, so a build for one vendor never
executes another vendor's code.

``MONAI_VENDOR`` selects a vendor explicitly: a registered name, ``none`` to disable rewriting, or
``auto`` (the default) to detect.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import re
import sys
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

__all__ = [
    "REGISTRY",
    "ActiveVendor",
    "VendorPluginError",
    "active_vendor",
    "apply_to_dependencies",
    "apply_to_optional_dependencies",
    "canonical_name",
    "load_by_path",
    "split_requirement",
]


class VendorPluginError(RuntimeError):
    """A vendor was detected but its plugin could not be loaded.

    Callers must not treat this as "no vendor": the hardware is known, so falling back to the
    unmodified lists would ship another vendor's packages in this vendor's wheel.
    """


def load_by_path(name: str, path: str) -> Any:
    """Import the module at ``path`` under ``name``, without importing its package.

    It is registered in ``sys.modules`` before execution because ``dataclasses`` resolves a class's
    module through it.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _torch_version_attr(attr: str) -> str:
    """``torch.version.<attr>`` as a string, or "" if torch is absent or has no such attribute."""
    try:
        import torch

        return getattr(torch.version, attr, None) or ""
    except Exception:  # torch absent, or present but not importable in this environment
        return ""


KFD_TOPOLOGY_NODES = "/sys/class/kfd/kfd/topology/nodes"


def _amd_gpu_present() -> bool:
    """True when the amdgpu driver exposes a GPU compute node. Node 0 is the host CPU."""
    try:
        nodes = sorted(os.scandir(KFD_TOPOLOGY_NODES), key=lambda node: node.name)
    except OSError:
        return False
    for node in nodes:
        try:
            with open(os.path.join(node.path, "properties")) as handle:
                text = handle.read()
        except OSError:
            continue
        match = re.search(r"^gfx_target_version (\d+)", text, re.M)
        if match and match.group(1) != "0":
            return True
    return False


@dataclass(frozen=True)
class _Registration:
    """A vendor plugin, the cheap probe deciding whether to load it, and an optional probe for the
    vendor's hardware used only to warn when that hardware is present but its PyTorch is not.
    """

    module: str
    detect: Callable[[], bool]
    hardware: Callable[[], bool] | None = None


# Probes must stay trivial and must not import the plugin module: they run on every build, including
# builds for other vendors. Add new vendors here and in the matching ``vendor_<name>.py``.
REGISTRY: dict[str, _Registration] = {
    "rocm": _Registration("vendor_rocm", lambda: bool(_torch_version_attr("hip")), _amd_gpu_present)
}

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


def _import_plugin(module: str) -> Any:
    """Import a sibling plugin module, whether this module was imported, run as a script or loaded
    by path from ``setup.py``.
    """
    if __package__:
        return importlib.import_module(f".{module}", __package__)
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{module}.py")
    return load_by_path(f"monai_{module}", path)


def _substitute(requirements: list[str], substitutions: dict[str, str | None]) -> list[str]:
    """Apply ``substitutions``, dropping None entries and de-duplicating the result."""
    out: list[str] = []
    for requirement in requirements:
        name, _, _ = split_requirement(requirement)
        key = canonical_name(name)
        if key in substitutions:
            replacement = substitutions[key]
            if replacement is None:
                continue
            requirement = replacement
        if requirement not in out:
            out.append(requirement)
    return out


def apply_to_dependencies(dependencies: list[str], plugin: Any) -> list[str]:
    """Rewrite required dependencies: substitute distributions, add the extras the plugin asks for
    and append its extra requirements.
    """
    out = []
    for requirement in _substitute(dependencies, plugin.SUBSTITUTIONS):
        name, extras, rest = split_requirement(requirement)
        added = plugin.extras_for(canonical_name(name))
        if added:
            requirement = f"{name}[{','.join(sorted(set(extras) | set(added)))}]{rest}"
        out.append(requirement)
    for requirement in plugin.extra_requirements():
        if requirement not in out:
            out.append(requirement)
    return out


def apply_to_optional_dependencies(optional_dependencies: dict[str, list[str]], plugin: Any) -> dict[str, list[str]]:
    """Rewrite each optional-dependency group, substituting distributions."""
    return {name: _substitute(list(group), plugin.SUBSTITUTIONS) for name, group in optional_dependencies.items()}


@dataclass(frozen=True)
class ActiveVendor:
    """The vendor plugin selected for this build, with the generic rewrites bound to it."""

    name: str
    plugin: Any

    def apply_to_dependencies(self, dependencies: list[str]) -> list[str]:
        return apply_to_dependencies(dependencies, self.plugin)

    def apply_to_optional_dependencies(self, optional_dependencies: dict[str, list[str]]) -> dict[str, list[str]]:
        return apply_to_optional_dependencies(optional_dependencies, self.plugin)


def _validate(name: str, plugin: Any) -> None:
    """Check the plugin contract before anything is rewritten with it.

    Neither mistake raises where it would be used. ``key in substitutions`` is a substring test on a
    string and a membership test on any container, so a ``SUBSTITUTIONS`` that is not a mapping
    quietly matches nothing; and lookups are canonicalised, so a key that is not already a PEP 503
    name can never match either. Both would ship another vendor's packages without a word.
    """
    substitutions = getattr(plugin, "SUBSTITUTIONS", None)
    if not isinstance(substitutions, Mapping):
        raise VendorPluginError(f"{name} plugin SUBSTITUTIONS must be a mapping")
    for key in substitutions:
        if not isinstance(key, str) or key != canonical_name(key):
            raise VendorPluginError(
                f"{name} plugin SUBSTITUTIONS key {key!r} is not a PEP 503 normalised name, "
                f"so it would never match a requirement"
            )
    for attribute in ("extras_for", "extra_requirements"):
        if not callable(getattr(plugin, attribute, None)):
            raise VendorPluginError(f"{name} plugin must define {attribute}()")


def _load(name: str) -> ActiveVendor:
    """Import the plugin for ``name``; a failure here is fatal, never a silent "no vendor"."""
    try:
        plugin = _import_plugin(REGISTRY[name].module)
    except Exception as exc:
        raise VendorPluginError(f"{name} was detected but its plugin could not be loaded: {exc}") from exc
    _validate(name, plugin)
    return ActiveVendor(name, plugin)


def active_vendor() -> ActiveVendor | None:
    """The vendor whose packages this build needs, or None to leave ``pyproject.toml`` alone."""
    requested = os.environ.get("MONAI_VENDOR", "").strip().lower()
    if requested in ("none", "cpu"):
        return None
    if requested and requested != "auto":
        if requested not in REGISTRY:
            raise ValueError(f"unknown MONAI_VENDOR {requested!r}; expected one of {sorted(REGISTRY)} or 'none'")
        return _load(requested)
    for name, registration in REGISTRY.items():
        if registration.detect():
            return _load(name)
    _warn_hardware_without_vendor()
    return None


def _warn_hardware_without_vendor() -> None:
    """Warn when a vendor's hardware is present but the torch this build sees is not its build.

    Under ``pip`` build isolation the build environment resolves ``torch`` afresh, and a generic
    PyPI wheel outranks a vendor build under PEP 440, so a source install on vendor hardware can
    otherwise emit another vendor's dependencies without a word.
    """
    for name, registration in REGISTRY.items():
        if registration.hardware is not None and registration.hardware():
            warnings.warn(
                f"{name} hardware is present, but the torch this build sees is not a {name} build, "
                f"so {name} dependencies were not applied. Build against a {name} torch (for a "
                f"source install, `pip install --no-build-isolation` in an environment that already "
                f"has one), or set MONAI_VENDOR={name} to apply them regardless.",
                stacklevel=3,
            )
            return

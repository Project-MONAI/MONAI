from importlib import import_module
from pkgutil import walk_packages
from re import match


def export(modname):
    """Make the decorated object a member of the named module."""

    def _inner(obj):
        mod = import_module(modname)
        if not hasattr(mod, obj.__name__):
            setattr(mod, obj.__name__, obj)

        return obj

    return _inner


def load_submodules(basemod, load_all=True, exclude_pattern="(.*[tT]est.*)|(_.*)"):
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

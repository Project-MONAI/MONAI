# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import subprocess
import sys
import importlib
import inspect

sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
print(sys.path)

import monai  # noqa: E402

# -- Project information -----------------------------------------------------
project = "MONAI"
copyright = "MONAI Consortium"
author = "MONAI Contributors"

# The full version, including alpha/beta/rc tags
short_version = monai.__version__.split("+")[0]
release = short_version
version = short_version

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "transforms",
    "networks",
    "metrics",
    "engines",
    "data",
    "apps",
    "fl",
    "bundle",
    "config",
    "handlers",
    "losses",
    "visualize",
    "utils",
    "inferers",
    "optimizers",
    "auto3dseg",
]


def generate_apidocs(*args):
    """Generate API docs automatically by trawling the available modules"""

    import pandas as pd
    from monai.bundle.properties import TrainProperties, InferProperties, MetaProperties

    csv_file = os.path.join(os.path.dirname(__file__), "train_properties.csv")  # used in mb_properties.rst
    pd.DataFrame.from_dict(TrainProperties, orient="index").iloc[:, :3].to_csv(csv_file)
    csv_file = os.path.join(os.path.dirname(__file__), "infer_properties.csv")
    pd.DataFrame.from_dict(InferProperties, orient="index").iloc[:, :3].to_csv(csv_file)
    csv_file = os.path.join(os.path.dirname(__file__), "meta_properties.csv")
    pd.DataFrame.from_dict(MetaProperties, orient="index").iloc[:, :3].to_csv(csv_file)

    module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "monai"))
    output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "apidocs"))
    apidoc_command_path = "sphinx-apidoc"
    if hasattr(sys, "real_prefix"):  # called from a virtualenv
        apidoc_command_path = os.path.join(sys.prefix, "bin", "sphinx-apidoc")
        apidoc_command_path = os.path.abspath(apidoc_command_path)
    print(f"output_path {output_path}")
    print(f"module_path {module_path}")
    subprocess.check_call(
        [apidoc_command_path, "-e"]
        + ["-o", output_path]
        + [module_path]
        + [os.path.join(module_path, p) for p in exclude_patterns]
    )


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
source_suffix = {".rst": "restructuredtext", ".txt": "restructuredtext", ".md": "markdown"}

extensions = [
    "recommonmark",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.linkcode",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.autosummary",
    "sphinx_autodoc_typehints",
]

autoclass_content = "class"
add_module_names = True
source_encoding = "utf-8"
autosectionlabel_prefix_document = True
napoleon_use_param = True
napoleon_include_init_with_doc = True
set_type_checking_flag = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"
# html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_theme_options = {
    "external_links": [{"url": "https://github.com/Project-MONAI/tutorials", "name": "Tutorials"}],
    "icon_links": [
        {"name": "GitHub", "url": "https://github.com/project-monai/monai", "icon": "fab fa-github-square"},
        {"name": "Twitter", "url": "https://twitter.com/projectmonai", "icon": "fab fa-twitter-square"},
    ],
    "collapse_navigation": True,
    "navigation_with_keys": True,
    "navigation_depth": 1,
    "show_toc_level": 1,
    "footer_start": ["copyright"],
    "navbar_align": "content",
    "logo": {"image_light": "MONAI-logo-color.png", "image_dark": "MONAI-logo-color.png"},
}
html_context = {
    "github_user": "Project-MONAI",
    "github_repo": "MONAI",
    "github_version": "dev",
    "doc_path": "docs/source",
    "conf_py_path": "/docs/source",
    "VERSION": version,
}
html_scaled_image_link = False
html_show_sourcelink = True
html_favicon = "../images/favicon.ico"
html_logo = "../images/MONAI-logo-color.png"
html_sidebars = {"**": ["search-field", "sidebar-nav-bs"]}
pygments_style = "sphinx"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["../_static"]
html_css_files = ["custom.css"]
html_title = f"{project} {version} Documentation"

# -- Auto-convert markdown pages to demo --------------------------------------


def setup(app):
    # Hook to allow for automatic generation of API docs
    # before doc deployment begins.
    app.connect("builder-inited", generate_apidocs)


# -- Linkcode configuration --------------------------------------------------
DEFAULT_REF = "dev"
read_the_docs_ref = os.environ.get("READTHEDOCS_GIT_IDENTIFIER", None)
if read_the_docs_ref:
    # When building on ReadTheDocs, link to the specific commit
    # https://docs.readthedocs.io/en/stable/reference/environment-variables.html#envvar-READTHEDOCS_GIT_IDENTIFIER
    git_ref = read_the_docs_ref
elif os.environ.get("GITHUB_REF_TYPE", "branch") == "tag":
    # When building a tag, link to the tag itself
    git_ref = os.environ.get("GITHUB_REF", DEFAULT_REF)
else:
    git_ref = os.environ.get("GITHUB_SHA", DEFAULT_REF)

DEFAULT_REPOSITORY = "Project-MONAI/MONAI"
repository = os.environ.get("GITHUB_REPOSITORY", DEFAULT_REPOSITORY)

base_code_url = f"https://github.com/{repository}/blob/{git_ref}"
MODULE_ROOT_FOLDER = "monai"
repo_root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


# Adjusted from https://github.com/python-websockets/websockets/blob/main/docs/conf.py
def linkcode_resolve(domain, info):
    if domain != "py":
        raise ValueError(
            f"expected domain to be 'py', got {domain}."
            "Please adjust linkcode_resolve to either handle this domain or ignore it."
        )

    mod = importlib.import_module(info["module"])
    if "." in info["fullname"]:
        objname, attrname = info["fullname"].split(".")
        obj = getattr(mod, objname)
        try:
            # object is a method of a class
            obj = getattr(obj, attrname)
        except AttributeError:
            # object is an attribute of a class
            return None
    else:
        obj = getattr(mod, info["fullname"])

    try:
        file = inspect.getsourcefile(obj)
        source, lineno = inspect.getsourcelines(obj)
    except TypeError:
        # e.g. object is a typing.Union
        return None
    file = os.path.relpath(file, repo_root_path)
    if not file.startswith(MODULE_ROOT_FOLDER):
        # e.g. object is a typing.NewType
        return None
    start, end = lineno, lineno + len(source) - 1
    url = f"{base_code_url}/{file}#L{start}-L{end}"
    return url

import os
import io
import re
from setuptools import setup, find_packages

# inspired by https://github.com/pytorch/ignite/blob/master/setup.py
def read(*names, **kwargs):
    print(os.path.join(os.path.dirname(__file__), *names))
    with io.open(
        os.path.join(os.path.dirname(__file__), *names), 
        encoding=kwargs.get("encoding", "utf8")
    ) as fp:
        return fp.read()

def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")

readme = read("README.md")
VERSION = find_version("monai", "__init__.py")
requirements = [
    "torch",
    "pytorch-ignite",
    "numpy",
    "pillow",
    "coverage",
    "nibabel",
    "parameterized",
    "tensorboard",
    "scikit-image",
    "scipy"
]

if __name__ == '__main__':
    setup(
        # Metadata
        name="monai",
        version=VERSION,
        author="MONAI Consortium",
        url="https://github.com/Project-MONAI/MONAI",
        author_email="monai.miccai2019@gmail.com",
        description="AI Toolkit for Healthcare Imaging",
        long_description_content_type="text/markdown",
        long_description=readme,
        license="Apache License 2.0",

        # Package info
        packages=find_packages(exclude=('docs', 'examples', 'tests')),
        zip_safe=True,
        install_requires=requirements
    )

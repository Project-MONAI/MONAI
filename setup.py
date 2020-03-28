import io
import os

from setuptools import find_packages, setup

import versioneer


# inspired by https://github.com/pytorch/ignite/blob/master/setup.py
def read(*names, **kwargs):
    print(os.path.join(os.path.dirname(__file__), *names))
    with io.open(os.path.join(os.path.dirname(__file__), *names), encoding=kwargs.get("encoding", "utf8")) as fp:
        return fp.read()


readme = read("README.md")
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
    "scipy",
]

if __name__ == '__main__':
    setup(
        # Metadata
        name="monai",
        version=versioneer.get_version(),
        cmdclass=versioneer.get_cmdclass().copy(),
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
        install_requires=requirements)

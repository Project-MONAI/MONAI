import sys
import os
import numpy as np
import setuptools
from setuptools import setup, find_packages
import glob
from glob import glob
import torch

from distutils.core import setup
from distutils.extension import Extension

_DEBUG = False

os.environ["CC"] = "clang"
os.environ["CXX"] = "clang"

package_name = 'GeodisTKcustom'
module_name  = 'GeodisTKcustom'
version      = sys.version[0]
extra_compile_args = []

#
# Installation note:
# 
# The include_dirs directive in the Extension() below need to have the include path to your Python "torch" module
# include file directory. (e.g. '/home/ala/.local/lib/python3.8/site-packages/torch/include/')
# in the include_dirs[] below is my WSL/Ubuntu torch include directory. It isn't a constant and it depends on your environment.
# Please replace the entry with your include path.
# You can use these commands to find your torch installation folder
# import torch
# print(torch.__file__)
#
module1 = Extension(module_name,
                    include_dirs = [np.get_include(),
                                    '/home/ala/.local/lib/python3.8/site-packages/torch/include/',
                                    './cpp'],
                    sources = ['./cpp/util.cpp', 
                               './cpp/geodesic_distance_2d.cpp', 
                               './cpp/geodesic_distance_3d.cpp', 
                               './cpp/geodesic_distance.cpp',
                               './cpp/geodesic_distance_pytorch.cpp'
                               ],
                    undef_macros=['DEBUG','_DEBUG'],
                    extra_link_args=['-mlinker-version=0']
)

# Get the summary
description = 'An open-source toolkit to calculate geodesic distance' + \
              ' for 2D and 3D images'

# Get the long description
if(sys.version[0] == '2'):
    import io
    with io.open('README.md', 'r', encoding='utf-8') as f:
        long_description = f.read()
else:
    with open('README.md', encoding='utf-8') as f:
        long_description = f.read()

setup(
    cmdclass={
        'ar': 'ld',
    },
    name    = package_name,
    version = "0.1.7",
    author  ='Ala Al-Afeef',
    author_email = 'ala.al-afeef@kcl.ac.uk',
    description  = description,
    long_description = long_description,
    long_description_content_type = 'text/markdown',
    url      = 'https://github.com/taigw/GeodisTK',
    license  = 'MIT',
    ext_modules = [module1],
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
    ],

    python_requires = '>=3.6',
    
)


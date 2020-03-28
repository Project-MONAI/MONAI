from setuptools import find_packages, setup

import versioneer

if __name__ == '__main__':
    setup(version=versioneer.get_version(),
          cmdclass=versioneer.get_cmdclass(),
          packages=find_packages(exclude=('docs', 'examples', 'tests')),
          zip_safe=True)

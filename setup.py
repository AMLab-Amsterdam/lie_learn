#pylint: disable=missing-docstring
import sys

from setuptools import dist, find_packages, setup

if sys.version_info[0] < 3:
    dist.Distribution().fetch_build_eggs(['Cython', 'numpy<1.17', 'requests'])
else:
    dist.Distribution().fetch_build_eggs(['Cython', 'numpy', 'requests'])

import numpy as np
from Cython.Build import cythonize

setup(
    name='lie_learn',
    packages=find_packages(),
    ext_modules=cythonize('lie_learn/**/*.pyx'),
    include_dirs=[np.get_include()],
    install_requires=[
        'cython',
        'requests',
    ],
    extras_require={
        ':python_version<"3.0"': ['scipy<1.3', 'numpy<1.17'],
        ':python_version>="3.0"': ['scipy', 'numpy'],
    },
)

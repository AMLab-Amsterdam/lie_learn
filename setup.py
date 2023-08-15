# pylint: disable=missing-docstring
import numpy as np
from Cython.Build import cythonize
from setuptools import setup, find_packages

setup(
    ext_modules=cythonize('lie_learn/**/*.pyx', language_level=2),
    include_dirs=[np.get_include()],
)

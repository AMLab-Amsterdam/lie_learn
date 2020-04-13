# pylint: disable=missing-docstring
import sys
import glob

from setuptools import dist, find_packages, setup, Extension

try:
    from Cython.Build import cythonize

    use_cython = True
except ImportError:
    use_cython = False

if sys.version_info[0] < 3:
    setup_requires_list = ['numpy<1.17']
else:
    setup_requires_list = ['numpy']

dist.Distribution().fetch_build_eggs(setup_requires_list)

import numpy as np

ext = '.pyx' if use_cython else '.c'
files = glob.glob('lie_learn/**/*' + ext, recursive=True)
extensions = [Extension(file.split('.')[0].replace('/', '.'), [file]) for file in files]
if use_cython:
    from Cython.Build import cythonize

    extensions = cythonize(extensions)

setup(
    name='lie_learn',
    version="0.0.1.post1",
    description="A python package that knows how to do various tricky computations related to Lie groups and "
                "manifolds (mainly the sphere S2 and rotation group SO3).",
    url="https://github.com/AMLab-Amsterdam/lie_learn",
    packages=find_packages(exclude=["tests.*", "tests"]),
    ext_modules=extensions,
    include_dirs=[np.get_include()],
    setup_requires=setup_requires_list,
    install_requires=[
        'requests',
        'numpy ; python_version>="3.0"',
        'scipy ; python_version>="3.0"',
        'numpy<1.17 ; python_version<"3.0"',
        'scipy<1.3 ; python_version<"3.0"',
    ],
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">2.7,!=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, !=3.4.*, !=3.5.*"
    # extras_require={
    #     "pynfft": ["pynfft"],  # This installation is complicated. Do it yourself.
    # }
)

# Usage:
# python setup.py build_ext --inplace

# build script for 'dvedit' - Python libdv wrapper

# change this as needed
#libdvIncludeDir = "/usr/include/libdv"

import glob, sys, os, stat, subprocess
from distutils.core import setup
from distutils.extension import Extension
import numpy as np

# we'd better have Cython installed, or it's a no-go
try:
    from Cython.Distutils import build_ext
except:
    print("You don't seem to have Cython installed. Please get a")
    print("copy from www.cython.org and install it")
    sys.exit(1)


# scan the 'dvedit' directory for extension files, converting
# them to extension names in dotted notation
def scandir(dir="./", files=[]):
    for file in os.listdir(dir):
        path = os.path.join(dir, file)
        if os.path.isfile(path) and path.endswith(".pyx"):
            # Exclude './' and '.pyx' and replace slashes with dots to get package name
            files.append(path.replace(os.path.sep, ".")[2:-4])
        elif os.path.isdir(path):
            scandir(path, files)
    return files


# generate an Extension object from its dotted name
def makeExtension(extName):
    extPath = extName.replace(".", os.path.sep) + ".pyx"
    return Extension(
        extName,
        [extPath],
        include_dirs = [np.get_include(), "."],   # adding the '.' to include_dirs is CRUCIAL!!
        extra_compile_args = ["-O3", "-Wall"],
        extra_link_args = ['-g'],
        libraries = [],
        )

def find_packages(root=".", excluded=[], prefix=""):
    packages = []
    package_dir = {}
    for p in glob.glob('{0}/**/__init__.py'.format(root), recursive=True):
        ps = p.split(os.path.sep)
        if not any([ p.startswith(q) for q in excluded ]):
            newp = ".".join([prefix] + ps[1:-1])
            packages.append(newp)
            package_dir[newp] = os.path.join(*ps[:-1])
    return packages, package_dir

# get the list of extensions
extNames = scandir("./")
print(extNames)

# and build up the set of Extension objects
extensions = [makeExtension(name) for name in extNames]
print(extensions)

packages, package_dir = find_packages(excluded=["./build"], prefix="lie_learn")
print(packages, package_dir)

# finally, we can pass all this to distutils
setup(
  name="lie_learn",
  packages=packages,
  package_dir=package_dir,
  ext_package="lie_learn",
  ext_modules=extensions,
  cmdclass = {'build_ext': build_ext},
)
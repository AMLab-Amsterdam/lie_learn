### This package is a fork of the original!
Original can be found at: [https://github.com/AMLab-Amsterdam/lie_learn](https://github.com/AMLab-Amsterdam/lie_learn)

lie_learn is a python package that knows how to do various tricky computations related to Lie groups and manifolds (mainly the sphere S2 and rotation group SO3). This package was written to support various machine learning projects, such as Harmonic Exponential Families [2], (continuous) Group Equivariant Networks [3], Steerable CNNs [4] and Spherical CNNs [5].

This code was developed using an extremely agile, move-fast-and-break-things, extreme-programming software development workflow, and was extensively tested using the print command. Most of the code was written in the 72 hours preceding conference deadlines. In other words, this code is a bit of a mess, but we're releasing it anyway because it could be useful to others.

# What this code can do
- Reparamterize rotations, e.g. matrix to Euler angles to quaternions, etc. (see groups & spaces modules)
- Compute the Wigner-d and Wigner-D matrices (the irreducible representations of SO(3)), and spherical harmonics, using the method developed by Pinchon & Hoggan [1] (see pinchon_hoggan_dense.py). This is a very fast and stable method, but requires a fairly large "J matrix", which we have precomputed up to order 278 using a Maple script. The code will automatically download it from Google Drive during installation.
Note: There are many normalization and phase conventions for both the real and complex versions of the D-matrices and spherical harmonics, and the code can convert between a lot of them (irrep_bases.pyx).
- Compute generalized / non-commutative FFTs for the sphere S2, rotation group SO3, and special Euclidean group SE2 (see spectral module).
- Fit Harmonic Exponential Families on the sphere (probability module; not sure code is still working)

# Installation
lie_learn can be installed from pypi using:

```
pip install lie_learn
```

Although cython is not a necessary dependency, if you have cython installed, cython will write new versions of the `*.c
` files before compiling them into `*.so` during installation. To use lie_learn, you will need a c compiler which is
 available to python setuptools.
 

# Feedback
For questions and comments, feel free to contact Taco Cohen (http://ta.co.nl).


# References
[1] Pinchon, D., & Hoggan, P. E. (2007). Rotation matrices for real spherical harmonics: general rotations of atomic orbitals in space-fixed axes. Journal of Physics A: Mathematical and Theoretical, 40(7), 1597–1610.

[2] Cohen, T. S., & Welling, M. (2015). Harmonic Exponential Families on Manifolds. In Proceedings of the 32nd International Conference on Machine Learning (ICML) (pp. 1757–1765).

[3] Cohen, T. S., & Welling, M. (2016). Group equivariant convolutional networks. In Proceedings of The 33rd International Conference on Machine Learning (ICML) (Vol. 48, pp. 2990–2999).

[4] Cohen, T. S., & Welling, M. (2017). Steerable CNNs. In ICLR.

[5] T.S. Cohen, M. Geiger, J. Koehler, M. Welling (2017). Convolutional Networks for Spherical Signals. In ICML Workshop on Principled Approaches to Deep Learning.

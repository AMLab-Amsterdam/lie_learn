
import numpy as np
from scipy.fftpack import fft, ifft, fftshift

from lie_learn.spectral.FFTBase import FFTBase
import lie_learn.spaces.S2 as S2
from lie_learn.representations.SO3.spherical_harmonics import csh, sh


class S2_FT_Naive(FFTBase):
    """
    The most naive implementation of the discrete spherical Fourier transform:
    explicitly construct the Fourier matrix F and multiply by it to perform the Fourier transform.
    """

    def __init__(self, L_max,
                 grid_type='Gauss-Legendre',
                 field='real', normalization='quantum', condon_shortley='cs'):

        super().__init__()

        self.b = L_max + 1

        # Compute a grid of spatial sampling points and associated quadrature weights
        theta, phi = S2.meshgrid(b=self.b, grid_type=grid_type)
        self.w = S2.quadrature_weights(b=self.b, grid_type=grid_type)
        self.spatial_grid_shape = theta.shape
        self.num_spatial_points = theta.size

        # Determine for which degree and order we want the spherical harmonics
        irreps = np.arange(self.b)  # TODO find out upper limit for exact integration for each grid type
        ls = [[ls] * (2 * ls + 1) for ls in irreps]
        ls = np.array([ll for sublist in ls for ll in sublist])  # 0, 1, 1, 1, 2, 2, 2, 2, 2, ...
        ms = [list(range(-ls, ls + 1)) for ls in irreps]
        ms = np.array([mm for sublist in ms for mm in sublist])  # 0, -1, 0, 1, -2, -1, 0, 1, 2, ...
        self.num_spectral_points = ms.size  # This equals sum_{l=0}^{b-1} 2l+1 = b^2

        # In one shot, sample the spherical harmonics at all spectral (l, m) and spatial (theta, phi) coordinates
        self.Y = sh(ls[None, None, :], ms[None, None, :], theta[:, :, None], phi[:, :, None],
                    field=field, normalization=normalization, condon_shortley=condon_shortley == 'cs')

        # Convert to a matrix
        self.Ymat = self.Y.reshape(self.num_spatial_points, self.num_spectral_points)

    def analyze(self, f):
        return self.Ymat.T.conj().dot((f * self.w).flatten())

    def synthesize(self, f_hat):
        return self.Ymat.dot(f_hat).reshape(self.spatial_grid_shape)


def setup_legendre_transform(b):
    """
    Compute a set of matrices containing coefficients to be used in a discrete Legendre transform.

    The Legendre transform of a data vector s[k] (k=0, ..., 2b-1) is defined as

    s_hat(l, m) = sum_{k=0}^{2b-1} P_l^m(cos(theta_k)) s[k]
    for l = 0, ..., b-1 and -l <= m <= l,
    where P_l^m is the associated Legendre function of degree l and order m,
    theta_k = ...

    Computing Fourier Transforms and Convolutions on the 2-Sphere
    J.R. Driscoll, D.M. Healy

    FFTs for the 2-Sphere–Improvements and Variations
    D.M. Healy, Jr., D.N. Rockmore, P.J. Kostelec, and S. Moore

    :param b:
    :return:
    """
    dim = np.sum(np.arange(b) * 2 + 1)
    lt = np.empty((dim, 2 * b))

    theta, _ = S2.linspace(b, grid_type='Driscoll-Healy')
    sample_points = np.cos(theta)

    # TODO move quadrature weight computation to S2.py
    weights = [(1. / b) * np.sin(np.pi * j * 0.5 / b) *
               np.sum([1. / (2 * l + 1) * np.sin((2 * l + 1) * np.pi * j * 0.5 / b)
                      for l in range(b)])
               for j in range(2 * b)]
    weights = np.array(weights)

    zeros = np.zeros_like(sample_points)
    i = 0
    for l in range(b):
        for m in range(-l, l + 1):
            # Z = np.sqrt(((2 * l + 1) * factorial(l - m)) / float(4 * np.pi * factorial(l + m))) * np.pi / 2
            # lt[i, :] = lpmv(m, l, sample_points) * weights * Z

            # The spherical harmonics code appears to be more stable than the (unnormalized) associated Legendre
            # function code.
            lt[i, :] = csh(l, m, theta, zeros, normalization='seismology').real * weights * np.pi / 2

            i += 1

    return lt


def sphere_fft(f, lt=None):
    """
    Compute the Spherical Fourier transform of f.
    We use complex, seismology-normalized, centered spherical harmonics, which are orthonormal (see rep_bases.py).

    The spherical Fourier transform is defined:
    \hat{f}_l^m = int_0^2pi dphi int_0^pi dtheta sin(theta) f(theta, phi) Y_l^{m*}(theta, phi)
    (where we use the invariant area element dOmega = sin(theta) dtheta dphi for the 2-sphere)

    Args:
      f: an array of shape (2B, 2B), where B is the bandwidth.

    Returns:
      f_hat: the Fourier transform of f. This is an ... shape array.

    Main source:
    Engineering Applications of Noncommutative Harmonic Analysis.
    4.7.2 - Orthogonal Expansions on the Sphere
    G.S. Chrikjian, A.B. Kyatkin

    Further information:
    SOFT: SO(3) Fourier Transforms
    Peter J. Kostelec and Daniel N. Rockmore

    Generalized FFTs-a survey of some recent results
    Maslen & Rockmore

    Computing Fourier transforms and convolutions on the 2-sphere.
    Driscoll, J., & Healy, D. (1994).

    :param f: array of samples of the function to be transformed. Shape (2 * b, 2 * b)
    :param lt: precomputed legendre transform matrices, from setup_legendre_transform().
    :return: f_hat, the spherical Fourier transform of f. This is an array of size sum_l=0^{b-1} 2 l + 1.
             the coefficients are ordered as (l=0, m=0), (l=1, m=-1), (l=1, m=0), (l=1,m=1), ...
    """

    assert f.shape[0] == f.shape[1]
    assert f.shape[0] % 2 == 0
    b = f.shape[0] // 2

    # First, FFT along the alpha axis (axis 0)
    F = fft(f, axis=0)

    if lt is None:
        lt = setup_legendre_transform(b)

    i = 0
    f_hat = np.zeros(np.sum(np.arange(b) * 2 + 1), dtype=complex)
    for l in range(b):
        for m in range(-l, l + 1):
            f_hat[i] = lt[i, :].dot(F[m, :])
            i += 1

    return f_hat
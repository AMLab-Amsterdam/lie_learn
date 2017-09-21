
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
        beta, alpha = S2.meshgrid(b=self.b, grid_type=grid_type)
        self.w = S2.quadrature_weights(b=self.b, grid_type=grid_type)
        self.spatial_grid_shape = beta.shape
        self.num_spatial_points = beta.size

        # Determine for which degree and order we want the spherical harmonics
        irreps = np.arange(self.b)  # TODO find out upper limit for exact integration for each grid type
        ls = [[ls] * (2 * ls + 1) for ls in irreps]
        ls = np.array([ll for sublist in ls for ll in sublist])  # 0, 1, 1, 1, 2, 2, 2, 2, 2, ...
        ms = [list(range(-ls, ls + 1)) for ls in irreps]
        ms = np.array([mm for sublist in ms for mm in sublist])  # 0, -1, 0, 1, -2, -1, 0, 1, 2, ...
        self.num_spectral_points = ms.size  # This equals sum_{l=0}^{b-1} 2l+1 = b^2

        # In one shot, sample the spherical harmonics at all spectral (l, m) and spatial (beta, alpha) coordinates
        self.Y = sh(ls[None, None, :], ms[None, None, :], beta[:, :, None], alpha[:, :, None],
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
    
    The discrete Legendre transform of a data vector s[k] (k=0, ..., 2b-1) is defined as

    s_hat(l, m) = sum_{k=0}^{2b-1} P_l^m(cos(beta_k)) s[k]
    for l = 0, ..., b-1 and -l <= m <= l,
    where P_l^m is the associated Legendre function of degree l and order m,
    beta_k = ...

    Computing Fourier Transforms and Convolutions on the 2-Sphere
    J.R. Driscoll, D.M. Healy

    FFTs for the 2-Sphere–Improvements and Variations
    D.M. Healy, Jr., D.N. Rockmore, P.J. Kostelec, and S. Moore

    :param b: bandwidth of the transform
    :return: lt, an array of shape (N, 2b), containing samples of the Legendre functions,
     where N is the number of spectral points for a signal of bandwidth b.
    """
    dim = np.sum(np.arange(b) * 2 + 1)
    lt = np.empty((2 * b, dim))

    beta, _ = S2.linspace(b, grid_type='Driscoll-Healy')
    sample_points = np.cos(beta)

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
            # (Note: the spherical harmonics evaluated at alpha=0 is the associated Legendre function))
            lt[:, i] = csh(l, m, beta, zeros, normalization='seismology').real * weights * np.pi / 2

            i += 1

    return lt


def setup_legendre_transform_indices(b):
    ms = [list(range(-ls, ls + 1)) for ls in range(b)]
    ms = [mm for sublist in ms for mm in sublist]  # 0, -1, 0, 1, -2, -1, 0, 1, 2, ...
    ms = [mm % (2 * b) for mm in ms]
    return ms


def sphere_fft(f, lt=None, lti=None):
    """
    Compute the Spherical Fourier transform of f.
    We use complex, seismology-normalized, centered spherical harmonics, which are orthonormal (see rep_bases.py).

    The spherical Fourier transform is defined:
    \hat{f}_l^m = int_0^pi dbeta sin(beta) int_0^2pi dalpha  f(beta, alpha) Y_l^{m*}(beta, alpha)
    (where we use the invariant area element dOmega = sin(beta) dbeta dalpha for the 2-sphere)
    
    We have Y_l^m(beta, alpha) = P_l^m(cos(beta)) * e^{im alpha}, where P_l^m is the associated Legendre function,
    so we can rewrite:
    \hat{f}_l^m = int_0^pi dbeta sin(beta) (int_0^2pi dalpha f(beta, alpha) e^{im alpha} ) P_l^m(cos(beta))
        
    The integral over alpha can be evaluated by FFT:
    \bar{f}(beta_k, m) = int_0^2pi dalpha f(beta_k, alpha) e^{im alpha} = FFT(f, axis=1)[beta_k, m]

    Then we have
    \hat{f}_l^m = int_0^pi sin(beta) dbeta  \bar{f}(beta, m) P_l^m(cos(beta))
                = sum_k \bar{f}[beta_k, m] P_l^m(cos(beta_k)) w_k
    For appropriate quadrature weights w_k. This sum is called the discrete Legendre transform of \bar{f}
    
    We return \hat{f} as a flat vector. Hence, the precomputed P_l^m(cos(beta_k)) w_k is stored as an array of with a 
    combined (l, m)-axis and a k axis. We bring the data \bar{f}[beta_k, m] into the same form, by indexing with lti
    and then reduce over the beta_k axis.
    
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

    :param f: array of samples of the function to be transformed. Shape (2 * b, 2 * b). grid_type: Driscoll-Healy
    :param lt: precomputed Legendre transform matrices, from setup_legendre_transform().
    :param lti: precomputed Legendre transform indices, from setup_legendre_transform_indices().
    :return: f_hat, the spherical Fourier transform of f. This is an array of size sum_l=0^{b-1} 2 l + 1.
             the coefficients are ordered as (l=0, m=0), (l=1, m=-1), (l=1, m=0), (l=1,m=1), ...
    """
    assert f.shape[-2] == f.shape[-1]
    assert f.shape[-2] % 2 == 0
    b = f.shape[-2] // 2

    if lt is None:
        lt = setup_legendre_transform(b)

    if lti is None:
        lti = setup_legendre_transform_indices(b)

    # First, FFT along the alpha axis (last axis)
    # This gives the array f_bar with axes for beta and m.
    f_bar = fft(f, axis=-1)

    # Perform Legendre transform
    f_hat = (f_bar[..., lti] * lt).sum(axis=-2)
    return f_hat

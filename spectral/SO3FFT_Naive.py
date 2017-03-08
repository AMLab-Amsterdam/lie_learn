
import numpy as np

from lie_learn.representations.SO3.irrep_bases import change_of_basis_matrix
from lie_learn.representations.SO3.pinchon_hoggan.pinchon_hoggan_dense import rot_mat, Jd

from .FFTBase import FFTBase
from scipy.fftpack import fft2, ifft2, fftshift

## TODO:
# Write testing code for these FFTs
# Write fast code for the real, quantum-normalized, centered / block-diagonal bases.

class SO3_FFT_NaiveComplex(FFTBase):

    def __init__(self, L_max, d=None, L2_normalized=True):

        if d is None:
            self.d = self.setup_d_transform(b=L_max + 1, J_dense=Jd, L2_normalized=L2_normalized)

        # TODO: compute quadrature weights for analyze() transform

    def analyze(self, f):
        raise NotImplementedError('SO3 analyze function not implemented yet')

    def synthesize(self, f_hat):
        """
        """
        b = len(self.d)

        # Perform the brute-force Legendre transform
        # Note: the frequencies where m=-B or n=-B are set to zero,
        # because they are not used in the forward transform either
        # (the forward transform is up to m=-l, l<B
        df_hat = [self.d[l] * f_hat[l][:, None, :] for l in range(b)]
        F = np.zeros((2 * b, 2 * b, 2 * b), dtype=complex)
        for l in range(b):
            F[b - l:b + l + 1, :,  b - l:b + l + 1] += df_hat[l]

        # The rest of the SO(3) FFT is just a standard torus FFT
        F = fftshift(F, axes=(0, 2))
        f = ifft2(F, axes=(0, 2))
        return f * (2 * b) ** 2


    @staticmethod
    def setup_d_transform(b, J_dense, L2_normalized):
        """

        """

        # We know how to efficiently compute d functions in the real basis using
        # Pinchon-Hoggans approach (d_real = J X(beta) J), but we want them in the
        # basis of complex centered spherical harmonics Y^{-l}, ..., Y^{l}.
        # These matrices perform that change of basis
        #C2R = [get_sh_change_of_basis(l,
        #                              frm=('complex', 'seismology', 'centered'),
        #                              to=('real', 'quantum', 'centered'))
        #       for l in range(b)]
        C2R = [change_of_basis_matrix(l,
                                      frm=('complex', 'seismology', 'centered', 'cs'),
                                      to=('real', 'quantum', 'centered', 'cs'))
               for l in range(b)]

        # Compute array of beta values as described in SOFT 2.0 documentation.
        beta = np.pi * (2 * np.arange(0, 2 * b) + 1) / (4. * b)

        # Compute d matrices in real basis (fast, stable),
        # then change them to complex basis (where the d-funcs are still real):
        d = [np.array([rot_mat(0, bt, 0, l, J_dense[l])
                       for bt in beta])
             for l in range(b)]
        d = [C2R[l].conj().T.dot(d[l]).dot(C2R[l]).real
             for l in range(b)]

        if L2_normalized:
            # The Unitary matrix elements have norm:
            # | U^\lambda_mn |^2 = 1/(2l+1)
            # where the 2-norm is defined in terms of normalized Haar measure.
            # So T = sqrt(2l + 1) U are L2-normalized functions
            d = [d[l] * np.sqrt(2 * l + 1) for l in range(len(d))]

        return d


class SO3_FFT_NaiveReal(FFTBase):

    def __init__(self, L_max, d=None, L2_normalized=True):

        self.L_max = L_max
        self.complex_fft = SO3_FFT_NaiveComplex(L_max=L_max, d=d, L2_normalized=L2_normalized)

        # Compute change of basis function:
        self.c2b = [change_of_basis_matrix(l,
                                           frm=('complex', 'seismology', 'centered', 'cs'),
                                           to=('real', 'quantum', 'centered', 'cs'))
                    for l in range(L_max + 1)]

    def analyze(self, f):
        raise NotImplementedError('SO3 analyze function not implemented yet')

    def synthesize(self, f_hat):
        """
        """
        # Change basis on f_hat
        # We have R = B * C * B.conj().T, where
        # B is the real-to-complex change of basis, C are the complex Wigner D functions,
        # and R are the real Wigner D functions.
        # We want to compute Tr(eta^T R) = Tr( (B.T * eta * B.conj())^T C)
        f_hat_complex = [self.c2b[l].T.dot(f_hat[l]).dot(self.c2b[l].conj()) for l in range(self.L_max + 1)]

        f = self.complex_fft.synthesize(f_hat_complex)
        return f.real


def SO3_fft(f, d=None, w=None):
    """
    Compute the complex SO(3) Fourier transform of f.

    The standard way to define the FT is:
    \hat{f}^l_mn = (2 J + 1)/(8 pi^2) *
       int_0^2pi da int_0^pi db sin(b) int_0^2pi dc f(a,b,c) D^{l*}_mn(a,b,c)

    The normalizing constant comes about because:
    int_SO(3) D^*(g) D(g) dg = 8 pi^2 / (2 J + 1)
    where D is any Wigner D function D^l_mn.

    This function computes the FT using the normalized D functions:
    \tilde{D} = 1/2pi sqrt((2J+1)/2) D
    where D are the rotation matrices in the basis of complex, seismology-normalized, centered spherical harmonics.

    Hence, this function computes:
    \hat{f}^l_mn = \int_SO(3) f(g) \tilde{D}^{l*}_mn(g) dg

    So that the FT of f = \tilde{D}^l_mn is 1 at (l,m,n) (and zero elsewhere).

    Args:
      f: an array of shape (2B, 2B, 2B), where B is the bandwidth.

    Returns:
      f_hat: the Fourier transform of f. A list of length B,
      where entry l contains an 2l+1 by 2l+1 array containing the projections
      of f onto matrix elements of the l-th irreducible representation of
      SO(3).

    Main source:
    SOFT: SO(3) Fourier Transforms
    Peter J. Kostelec and Daniel N. Rockmore

    Further information:
    Generalized FFTs-a survey of some recent results
    Maslen & Rockmore

    Engineering Applications of Noncommutative Harmonic Analysis.
    9.5 - Sampling and FFT for SO(3) and SU(2)
    G.S. Chrikjian, A.B. Kyatkin
    """
    assert f.shape[0] == f.shape[1]
    assert f.shape[1] == f.shape[2]
    assert f.shape[0] % 2 == 0
    b = f.shape[0] / 2

    # First, FFT along the alpha and gamma axes (axis 0 and 2, respectively)
    # We use the inverse FFT (ifft) here, because the np.fft sign convention
    # for alpha and gamma is reversed relative to our D functions.
    #F = np.fft.ifft2(f.conj(), axes=(0, 2)).conj()
    F = fft2(f, axes=(0, 2))
    F = fftshift(F, axes=(0, 2))
    f0 = F.shape[0] / 2  # The index of the 0-frequency / DC component

    if d is None:
        # TODO: compute weights separately.
        d = setup_d_transform(b, weight=True)

    f_hat = []
    Z = 2 * np.pi / ((2 * b) ** 2)  # Normalizing constant
    for l in range(b):
        # Dot the vectors F_mn and d_mn over the middle axis (beta),
        # where -l <= m,n <= l, which corresponds to
        # f0 - l <= m,n <= f0 + l + 1
        # for 0-based indexing and zero-frequency location f0
        f_hat.append(np.sum(d[l] * F[f0 - l:f0 + l + 1, :, f0 - l:f0 + l + 1],
                            axis=1) * Z)
    return f_hat


def SO3_ifft(f_hat, d):
    """
    """
    b = len(d)
    df_hat = [d[l] * f_hat[l][:, None, :] for l in range(len(d))]

    # Note: the frequencies where m=-B or n=-B are set to zero,
    # because they are not used in the forward transform either
    # (the forward transform is up to m=-l, l<B
    F = np.zeros((2 * b, 2 * b, 2 * b), dtype=complex)
    for l in range(b):
        F[b - l:b + l + 1, :,  b - l:b + l + 1] += df_hat[l]

    F = fftshift(F, axes=(0, 2))
    f = ifft2(F, axes=(0, 2))
    return f * 2 * (b ** 2) / np.pi


def setup_d_transform(b, J_dense):
    """

    """

    # We know how to efficiently compute d functions in the real basis using
    # Pinchon-Hoggans approach (d_real = J X(beta) J), but we want them in the
    # basis of complex centered spherical harmonics Y^{-l}, ..., Y^{l}.
    # These matrices perform that change of basis
    ##C2R = [cc2rcph(l) for l in range(b)]
    #C2R = [get_sh_change_of_basis(l,
    #                              frm=('complex', 'seismology', 'centered'),
    #                              to=('real', 'quantum', 'centered'))
    #       for l in range(b)]
    C2R = [change_of_basis_matrix(l,
                                  frm=('complex', 'seismology', 'centered', 'cs'),
                                  to=('real', 'quantum', 'centered', 'cs'))
           for l in range(b)]

    # Compute array of beta values as described in SOFT 2.0 documentation.
    beta = np.pi * (2 * np.arange(0, 2 * b) + 1) / (4. * b)

    # Compute d matrices in real basis (fast, stable),
    # then change them to complex basis (where the d-funcs are still real):
    d = [np.array([rot_mat(0, bt, 0, l, J_dense[l])
                   for bt in beta])
         for l in range(b)]
    d = [C2R[l].conj().T.dot(d[l]).dot(C2R[l]).real
         for l in range(b)]
    #d = [np.array([wignerd_mat(l, b, approx_lim=100000)
    #               for b in beta]).transpose(1, 0, 2)
    #     for l in range(B)]

    # We want the L2 normalized functions:
    d = [d[l] * np.sqrt(l + 0.5) for l in range(len(d))]

    # Construct quadrature weights w
    # NO: this should be separate so that we can use the same d for forward and backward transforms
    """if weight:
        k = np.arange(0, b)
        w = np.array([(2. / b) * np.sin(np.pi * (2. * j + 1.) / (4. * b)) *
                      (np.sum((1. / (2 * k + 1))
                              * np.sin((2 * j + 1) * (2 * k + 1)
                                       * np.pi / (4. * b))))
                      for j in range(2 * b)])
        # In the SOFT source, they talk about the following weights being used for
        # odd-order transforms. Do not understand this, and the weights used above
        # (defined in the SOFT papers) seems to work.
        #w = np.array([(2. / B) *
        #              (np.sum((1. / (2 * k + 1))
        #                      * np.sin((2 * j + 1) * (2 * k + 1)
        #                               * np.pi / (4. * B))))
        #              for j in range(2 * B)])

        # Apply quadrature weights to the wigner d function samples:
        d = [d[l] * w[None, :, None] for l in range(b)]
    """
    return d


def SO3_convolve(f, g, dw=None, d=None):

    assert f.shape == g.shape
    assert f.shape[0] % 2 == 0
    b = f.shape[0] / 2

    if d is None:
        d = setup_d_transform(b)

    # To convolve, first perform a Fourier transform on f and g:
    F = SO3_fft(f, dw)
    G = SO3_fft(g, dw)

    # The Fourier transform of the convolution f*g is the matrix product FG
    # of their Fourier transforms F and G:
    FG = [np.dot(a, b) for (a, b) in zip(F, G)]

    # The convolution is obtain by inverse Fourier transforming FG:
    return SO3_ifft(FG, d)

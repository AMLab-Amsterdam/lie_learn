
import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from .FFTBase import FFTBase


class T2FFT(FFTBase):
    """
    The Fast Fourier Transform on the 2-Torus.

    REMOVE?

    The torus is parameterized by two cyclic variables (x, y).
    The standard domain is (x, y) in [0, 1) x [0, 1), in which case the Fourier basis functions are:
     exp( i 2 pi xi^T (x; y))
    where xi is the spectral variable, xi in Z^2.

    The Fourier transform is
    \hat{f}[p, q] = 1/2pi int_0^2pi f(x, y) exp(-i 2 pi xi^T (x; y)) dx dy



    but this class allows one to use arbitrarily scaled and shifted domains D = [l_x, u_x) x [l_y, u_y)
    Let the width of the domain be given by
      alpha_x = u_x - l_x
      alpha_y = u_y - l_y
    The basis functions on [l_x, u_x) x [l_y, u_y) are
     exp( i 2 pi xi^T ((x - l_x) / alpha_x; (y - l_y) / alpha_y))
    where xi is the spectral variable, xi in Z^2.
    The normalized Haar measure is dx dy / (alpha_x * alpha_y) (in terms of Lebesque measure dx dy)

    So the Fourier transform on this particular parameterization of the torus is:
    \hat{f}_pq = 1/alpha int_lx^ux int_ly^uy f(x) e^{-2 pi i (p, q)^T ((x - lx) / alpha_x; (y - ly)/alpha_y)} dx dy

    This is what the current class computes, given discrete samples in the domain D.
    The samples are assumed to come from the following sampling grid:
    (x_i, y_j), i = 0, ... N - 1; j = 0, ..., N - 1
    x_i = lx + alpha_x * (i / N_x)
    y_i = ly + alpha_y * (i / N_y)
    this is the ouput of
    x = np.linspace(lx, ux, N_x, endpoint=False)
    x = np.linspace(ly, uy, N_y, endpoint=False)
    X, Y = np.meshgrid(x, y)

    """
    def __init__(self, lower_bound=(0., 0.), upper_bound=(1., 1.)):
        self.lower_bound = np.array(lower_bound)
        self.upper_bound = np.array(upper_bound)


    @staticmethod
    def analyze(f, axes=(0, 1)):
        """
        Compute the Fourier Transform of the discretely sampled function f : T^2 -> C.

        Let f : T^2 -> C be a band-limited function on the torus.
        The samples f(theta_k, phi_l) correspond to points on a regular grid on the circle,
        as returned by spaces.T1.linspace:
        theta_k = phi_k = 2 pi k / N
        for k = 0, ..., N - 1 and l = 0, ..., N - 1

        This function computes
        \hat{f}_n = (1/N) \sum_{k=0}^{N-1} f(theta_k) e^{-i n theta_k}
        which, if f has band-limit less than N, is equal to:
        \hat{f}_n = \int_0^{2pi} f(theta) e^{-i n theta} dtheta / 2pi,
                  = <f(theta), e^{i n theta}>
        where dtheta / 2pi is the normalized Haar measure on T^1, and < , > denotes the inner product on Hilbert space,
        with respect to which this transform is unitary.

        The range of frequencies n is -floor(N/2) <= n <= ceil(N/2) - 1

        :param f:
        :param axis:
        :return:
        """
        # The numpy FFT returns coefficients in a different order than we want them,
        # and using a different normalization.
        f_hat = fft2(f, axes=axes)
        f_hat = fftshift(f_hat, axes=axes)
        size = np.prod([f.shape[ax] for ax in axes])
        return f_hat / size

    @staticmethod
    def synthesize(f_hat, axes=(0, 1)):
        """
        :param f_hat:
        :param axis:
        :return:
        """

        size = np.prod([f_hat.shape[ax] for ax in axes])
        f_hat = ifftshift(f_hat * size, axes=axes)
        f = ifft2(f_hat, axes=axes)
        return f

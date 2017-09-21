
import numpy as np
from numpy.fft import fft, ifft, fftshift, ifftshift
from .FFTBase import FFTBase


class T1FFT(FFTBase):
    """
    The Fast Fourier Transform on the Circle / 1-Torus / 1-Sphere.
    """

    @staticmethod
    def analyze(f, axis=0):
        """
        Compute the Fourier Transform of the discretely sampled function f : T^1 -> C.

        Let f : T^1 -> C be a band-limited function on the circle.
        The samples f(theta_k) correspond to points on a regular grid on the circle, as returned by spaces.T1.linspace:
        theta_k = 2 pi k / N
        for k = 0, ..., N - 1

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
        fhat = fft(f, axis=axis)
        fhat = fftshift(fhat, axes=axis)
        return fhat / f.shape[axis]

    @staticmethod
    def synthesize(f_hat, axis=0):
        """
        Compute the inverse / synthesis Fourier transform of the function f_hat : Z -> C.
        The function f_hat(n) is sampled at points in a limited range -floor(N/2) <= n <= ceil(N/2) - 1

        This function returns
        f[k] = f(theta_k) = sum_{n=-floor(N/2)}^{ceil(N/2)-1} f_hat(n) exp(i n theta_k)
        where theta_k = 2 pi k / N
        for k = 0, ..., N - 1

        :param f_hat:
        :param axis:
        :return:
        """

        f_hat = ifftshift(f_hat * f_hat.shape[axis], axes=axis)
        f = ifft(f_hat, axis=axis)
        return f

    @staticmethod
    def analyze_naive(f):
        f_hat = np.zeros_like(f)
        for n in range(f.size):
            for k in range(f.size):
                theta_k = k * 2 * np.pi / f.size
                f_hat[n] += f[k] * np.exp(-1j * n * theta_k)
        return fftshift(f_hat / f.size, axes=0)

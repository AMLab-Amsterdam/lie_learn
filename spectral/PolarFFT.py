
import numpy as np
from .FFTBase import FFTBase
from pynfft.nfft import NFFT


# UNFINISHED

class PolarFFT(FFTBase):

    def __init__(self, nx, ny, nt, nr):

        # Initialize the non-equispaced FFT
        self.nfft = NFFT(N=(nx, ny), M=nx * ny, n=None, m=12, flags=None)

        # Set up the polar sampling grid
        theta = np.linspace(0, 2 * np.pi, nt)
        r = np.linspace(0, 1., nr)
        T, R = np.meshgrid(theta, r)
        self.nfft.x = np.c_[T[..., None], R[..., None]].flatten()
        self.nfft.precompute()

    def analyze(self, f):

        self.nfft.f_hat = f
        f_hat = self.nfft.forward()

        return f_hat

    def synthesize(self, f_hat):

        self.nfft.f = f_hat
        f = self.nfft.adjoint()

        return f_hat

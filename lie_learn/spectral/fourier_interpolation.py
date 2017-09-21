
import numpy as np
from pynfft import nfft
from pynfft.solver import Solver

from .T2FFT import T2FFT


class FourierInterpolator(object):

    def __init__(self, cartesian_grid_shape, nonequispaced_grid):
        """
        The FourierInterpolator can interpolate data on an equispaced Cartesian grid to a non-equispaced grid.
        The inpterpolation works by first computing the Fourier coefficients of the input grid, and then evaluating
        the Fourier series defined by those coefficients at the non-equispaced grid.
        This operation is exactly invertible, as long as the Fourier coefficients are recoverable
        from the non-equispaced output samples.

        :param cartesian_grid_shape: the shape (nx, ny) of the input grid.
           Samples are assumed to be in [-.5, .5) x [-.5, .5)
        :param nonequispaced_grid: the output grid points. Shape (M, 2)
        """
        self.cartesian_grid_shape = cartesian_grid_shape
        self.nonequispaced_grid_shape = nonequispaced_grid.shape[:-1]
        self.nonequispaced_grid = nonequispaced_grid.reshape(-1, 2)
        self.nfft = nfft.NFFT(N=cartesian_grid_shape, M=np.prod(nonequispaced_grid.shape[:-1]),
                              n=None, m=12, flags=None)
        self.nfft.x = self.nonequispaced_grid
        self.nfft.precompute()
        self.solver = Solver(self.nfft)


    @staticmethod
    def init_cartesian_to_polar(nr, nt, nx, ny):

        # On the computation of the polar FFT
        # Markus Fenn, Stefan Kunis, Daniel Potts
        r = np.linspace(0, 1. / np.sqrt(2), nr)  # radius = sqrt((0 - 0.5)^2 + (0 - 0.5)^2) = sqrt(0.5) = 1/sqrt(2)
        t = np.linspace(0, 2 * np.pi, nt, endpoint=False)
        R, T = np.meshgrid(r, t, indexing='ij')
        X = R * np.cos(T)
        Y = R * np.sin(T)
        C = np.c_[X[..., None], Y[..., None]]
        return FourierInterpolator(cartesian_grid_shape=(nx, ny), nonequispaced_grid=C)

    def forward(self, f):
        """
        :param f:
        :return:
        """

        # Fourier transform x:
        # Perform a regular FFT:
        f_hat = T2FFT.analyze(f)

        print(f_hat)

        # Since this equispaced FFT assumes spatial samples in theta_k in [0, 1)
        # [assuming basis functions exp(i 2 pi n theta), not exp(i n theta)],
        # we shift by 0.5, i.e. multiply by exp(-i pi n) = (-1)^n
        f_hat *= ((-1) ** np.arange(-np.floor(f.shape[0] / 2), np.ceil(f.shape[0] / 2)))[:, None]
        f_hat *= ((-1) ** np.arange(-np.floor(f.shape[1] / 2), np.ceil(f.shape[1] / 2)))[None, :]

        print(f_hat)

        # Use NFFT to evaluate the function defined by these Fourier coefficients at the non-equispaced output grid
        self.nfft.f_hat = f_hat
        f_resampled = self.nfft.trafo().reshape(self.nonequispaced_grid_shape).copy()

        print(f_resampled)
        return f_resampled

    def backward(self, f):
        """

        :param f:
        :return:
        """

        self.solver.y = f
        self.solver.before_loop()
        for i in range(40):
            self.solver.loop_one_step()

        f_hat = self.solver.f_hat_iter

        # Since this equispaced FFT assumes spatial samples in theta_k in [0, 1)
        # [assuming basis functions exp(i 2 pi n theta), not exp(i n theta)],
        # we shift by 0.5, i.e. multiply by exp(-i pi n) = (-1)^n
        f_hat /= ((-1) ** np.arange(-np.floor(f_hat.shape[0] / 2), np.ceil(f_hat.shape[0] / 2)))[:, None]
        f_hat /= ((-1) ** np.arange(-np.floor(f_hat.shape[1] / 2), np.ceil(f_hat.shape[1] / 2)))[None, :]
        f = T2FFT.synthesize(f_hat)

        return f



def test2():

    nr = 100
    nt = 100
    nx = 20
    ny = 20

    F = FourierInterpolator.init_cartesian_to_polar(nr=nr, nt=nt, nx=nx, ny=ny)

    X, Y = np.meshgrid(np.linspace(-0.5, 0.5, nx, endpoint=False), np.linspace(-0.5, 0.5, ny, endpoint=False),
                       indexing='ij')
    f = np.exp(2*np.pi*1j*(X+0.5))  # + np.exp(2*np.pi * 1j * (3*(Y+0.5)))
    C = np.c_[X[..., None], Y[..., None]].reshape(-1, 2)

    #F = FourierInterpolator(grid_in_shape=X.shape, grid_out=C)
    print('aa')
    fp = F.forward(f)

    fr = F.backward(fp)
    return F, f, fp, fr


def test(sx=0, sy=0):

    nx = 33
    ny = 37
    nt = 16
    nr = 16
    f = np.zeros((nx, ny), dtype='complex')

    F = nfft.NFFT(N=(nx, ny), M=nx * ny)
    X, Y = np.meshgrid(np.linspace(-0.5, 0.5, nx, endpoint=False), np.linspace(-0.5, 0.5, ny, endpoint=False),
                       indexing='ij')
    f = np.exp(2*np.pi*1j*(X+0.5))
    F.x = np.c_[X[..., None], Y[..., None]].reshape(-1, 2)
    F.precompute()

    f_hat = T2FFT.analyze(f)
    tf_hat = f_hat.copy()
    tf_hat *= np.exp((2. * np.pi * 1j * sx * np.arange(-np.floor(f.shape[0] / 2.), np.ceil(f.shape[0] / 2.))[:, None]) / f.shape[0])
    tf_hat *= np.exp((2. * np.pi * 1j * sy * np.arange(-np.floor(f.shape[1] / 2.), np.ceil(f.shape[1] / 2.))[None, :]) / f.shape[1])


    F.f_hat = f_hat.conj()
    f_reconst1 = F.trafo().copy().conj()

    F.f_hat = tf_hat.conj()
    f_reconst2 = F.trafo().copy().conj()

    return F, f, f_hat, tf_hat, f_reconst1, f_reconst2
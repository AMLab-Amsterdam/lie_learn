

import numpy as np
# from numpy.fft import fft, fft2, ifft, ifft2, fftshift
from spectral.T1FFT import T1FFT
from spectral.T2FFT import T2FFT
from scipy.ndimage.interpolation import map_coordinates

from spectral.FFTBase import FFTBase
from spectral.fourier_interpolation import FourierInterpolator

import groups.SE2 as SE2

def bilinear_interpolate(f, x, y):
    x = np.asarray(x)
    y = np.asarray(y)

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, f.shape[1] - 1)
    x1 = np.clip(x1, 0, f.shape[1] - 1)
    y0 = np.clip(y0, 0, f.shape[0] - 1)
    y1 = np.clip(y1, 0, f.shape[0] - 1)

    Ia = f[y0, x0]
    Ib = f[y1, x0]
    Ic = f[y0, x1]
    Id = f[y1, x1]

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    print(x0.shape, y0.shape, x1.shape, y1.shape, x.shape, y.shape)
    print(Ia.shape, Ib.shape, Ic.shape, Id.shape)
    print(wa.shape, wb.shape, wc.shape, wd.shape)

    return wa[..., None] * Ia + wb[..., None] * Ib + wc[..., None] * Ic + wd[..., None] * Id


def mul(fh1, fh2):

    assert fh1.shape == fh2.shape

    # The axes of fh are (r, p, q)
    # For each r, we multiply the infinite dimensional matrices indexed by (p, q), assuming the values are zero
    # outside the range stored.
    # Thus, the p-axis of the second array fh2 must be truncated at both sides so that we can compute fh1.dot(fh2),
    # and so that the 0-frequency q-component of fh1 lines up with the zero-fruency p-component of fh2.
    p0 = fh1.shape[1] // 2  # Indices of the zero frequency component
    q0 = fh1.shape[2] // 2

    # The lower and upper bound of the p-range
    a = p0 - q0
    b = p0 + np.ceil(fh2.shape[2] / 2.)

    fh12 = []
    for i in range(fh1.shape[0]):

        fh12.append(fh1[i, :, :].dot(fh2[i, a:b, :]))

    fh12 = np.c_[fh12]  #.transpose(2, 0, 1)
    return fh12

def mulT(fh1, fh2):

    assert fh1.shape == fh2.shape

    # The axes of fh are (r, p, q)
    # For each r, we multiply the infinite dimensional matrices indexed by (p, q), assuming the values are zero
    # outside the range stored.
    # Thus, the p-axis of the second array fh2 must be truncated at both sides so that we can compute fh1.dot(fh2),
    # and so that the 0-frequency q-component of fh1 lines up with the zero-fruency p-component of fh2.
    p0 = fh1.shape[1] // 2  # Indices of the zero frequency component
    q0 = fh1.shape[2] // 2

    # The lower and upper bound of the p-range
    a = p0 - q0
    b = p0 + np.ceil(fh2.shape[2] / 2.)

    fh12 = []
    for i in range(fh1.shape[0]):

        fh12.append(fh1[i, :, :].dot(fh2[i, :, :].T)[:, a:b])

    fh12 = np.c_[fh12]  #.transpose(2, 0, 1)
    return fh12


def conv_test():

    f, f1c, f1p, f2, f2f, fh, fi, f1ci, f1pi, f2i, f2fi, fhi = test()
    fh2 = mulT(fh, fh)

    F = SE2_FFT(spatial_grid_size=(40,40,42), interpolation_method='spline', oversampling_factor=5)
    fi, f1ci, f1pi, f2i, f2fi, fhi = F.synthesize(fh2)
    from utils.visualize import plotmat
    for i in range(fi.shape[0]):
        plotmat(fi[:, :, i].real, i, range=(np.min(fi.real), np.max(fi.real)))

def SE2_matrix_element(r, p, q, tau, theta):
    from scipy.special import jv
    a = np.sqrt(tau[0] ** 2 + tau[1] ** 2)
    phi = np.angle(z=tau[0] + 1j * tau[1], deg=0)
    return 1j ** (q - p) * np.exp(1j * ((p - q) * phi + q * theta)) * jv(p - q, r * a)

def SE2_matrix_element_grid(r, p, q, spatial_grid_size=(10, 10, 10)):

    mat = np.zeros(spatial_grid_size, dtype='complex')
    taus = np.linspace(-1, 1, spatial_grid_size[0])
    thetas = np.linspace(0, 2 * np.pi, spatial_grid_size[2])
    for itau1 in range(spatial_grid_size[0]):
        for itau2 in range(spatial_grid_size[1]):
            for itheta in range(spatial_grid_size[2]):
                mat[itau1, itau2, itheta] = SE2_matrix_element(r, p, q, tau=(taus[itau1], taus[itau2]), theta=thetas[itheta])
    return mat


def SE2_matrix_element_chirkijian(r, p, q, tau, theta):
    # this appears to be the complex conjugate of what I derived
    from scipy.special import jv
    # Should compute SE2 matrix elements, by eq. 10.3 of Chirikjian & Kyatkin. Not yet tested
    a = np.sqrt(tau[0] ** 2 + tau[1] ** 2)
    phi = np.angle(z=tau[0] + 1j * tau[1], deg=0)
    return 1j ** (q - p) * np.exp(-1j * (q * theta + (p - q) * phi)) * jv(q - p, r * a)


def pix_to_ndc(C, w, h, flip_y=True):

    Xpix = C[..., 0]
    Ypix = C[..., 1]

    Xndc = (2. * Xpix - w) / w
    Yndc = (2. * Ypix - h) / h * (-1) ** flip_y
    return np.c_[Xndc[..., None], Yndc[..., None]]

def ndc_to_pix(C, w, h, flip_y=True):

    Xndc = C[..., 0]
    Yndc = C[..., 1] * (-1) ** flip_y

    Xpix = (Xndc + 1) * 0.5 * w
    Ypix = (Yndc + 1) * 0.5 * h
    return np.c_[Xpix[..., None], Ypix[..., None]]


def R2_SE2_convolve_naive(f1, f2, t_res=21, r_res=21, f_res=None):

    # Compute int_R2 f1(x) f2(x^{-1} g) dx = int_R2 f1(gx) f2(x^{-1}) dx
    w = f1.shape[0] - 1; h = f1.shape[1] - 1
    if f_res is None:
        f_res = f1.shape[0]

    # make a coordinate grid
    X, Y = np.meshgrid(np.linspace(-1, 1, f_res), np.linspace(-1, 1, f_res), indexing='ij')
    C = np.c_[X[..., None], Y[..., None]]

    # Create a flipped image f2(x^{-1})
    Xi, Yi = np.meshgrid(np.linspace(1, -1, f1.shape[0]), np.linspace(1, -1, f1.shape[1]), indexing='ij')
    Cinv = np.c_[X[..., None], Y[..., None]]
    Cinv_pix = ndc_to_pix(Cinv, w=w, h=h)
    f2inv = map_coordinates(f2, Cinv_pix.transpose(2, 0, 1), order=0, mode='constant', cval=0.0)

    out = np.empty((r_res, t_res, t_res))
    translations = np.linspace(-1, 1, t_res)
    rotations = np.linspace(0, 2 * np.pi, r_res, endpoint=False)
    for t1i in range(translations.size):
        for t2i in range(translations.size):
            for thetai in range(rotations.size):
                t1 = translations[t1i]
                t2 = translations[t2i]
                theta = rotations[thetai]

                # Transform the sampling grid:
                gC = SE2.transform(g=(theta, t1, t2), g_parameterization='rotation-translation',
                                   x=C, x_parameterization='cartesian')

                # Map normalized device coordinates to array indices
                gC_pix = ndc_to_pix(gC, w=w, h=h)

                # Evaluate f1 at the transformed grid:
                f1_gx = map_coordinates(f1, order=1, coordinates=gC_pix.transpose(2, 0, 1), mode='constant', cval=0.0)

                # Compute dot product:
                out[thetai, t1i, t2i] = (f1_gx * f2inv).sum()

    return out


def map_wrap(f, coords):

    # Create an agumented array, where the last row and column are added at the beginning of the axes
    fa = np.empty((f.shape[0] + 1, f.shape[1] + 1))
    #fa[1:, 1:] = f
    #fa[0, 1:] = f[-1, :]
    #fa[1:, 0] = f[:, -1]
    #f[0, 0] = f[-1, -1]
    fa[:-1, :-1] = f
    fa[-1, :-1] = f[0, :]
    fa[:-1, -1] = f[:, 0]
    fa[-1, -1] = f[0, 0]

    # Wrap coordinates
    wrapped_coords_x = coords[0, ...] % f.shape[0]
    wrapped_coords_y = coords[1, ...] % f.shape[1]
    wrapped_coords = np.r_[wrapped_coords_x[None, ...], wrapped_coords_y[None, ...]]

    # Interpolate
    #return fa, wrapped_coords, map_coordinates(f, wrapped_coords, order=1, mode='constant', cval=np.nan, prefilter=False)
    return map_coordinates(fa, wrapped_coords, order=1, mode='constant', cval=np.nan, prefilter=False)



def test():

    f = np.zeros((40, 40, 42))
    f[19:21, 10:30, :] = 1.

    F = SE2_FFT(spatial_grid_size=(40, 40, 42),
                interpolation_method='spline',
                spline_order=1,
                oversampling_factor=5)

    f, f1c, f1p, f2, f2f, fh = F.analyze(f)

    fi, f1ci, f1pi, f2i, f2fi, fhi = F.synthesize(fh)

    print(np.sum(np.abs(f - fi)))
    return f, f1c, f1p, f2, f2f, fh, fi, f1ci, f1pi, f2i, f2fi, fhi


def test_phaseshift1():

    nx = 20
    ny = 20
    p0 = nx / 2
    q0 = ny / 2

    # Shows that when the image rotates around center (p0, q0), the FT also rotates around (p0, q0) (which corresponds
    # to frequency (0, 0).
    f1 = np.zeros((nx, ny))
    f1[p0 - 1, q0 - 1] = 1.
    f1[p0, q0] = 1.
    f1[p0 + 1, p0 + 1] = 1.

    f1 = np.random.randn(nx, ny) + 1j * np.random.randn(nx, ny)

    X, Y = np.meshgrid(np.arange(p0, p0 + f1.shape[0]) % f1.shape[0],
                       np.arange(q0, q0 + f1.shape[1]) % f1.shape[1],
                       indexing='ij')
    f1shift = f1[X, Y]

    f1h = T2FFT.analyze(f1)
    f1sh = T2FFT.analyze(f1shift)

    # Do a phase shift and check that it is equal to the FT of the shifted image
    delta = -0.5  # we're shifting from [0, 1) to [-0.5, 0.5)
    xi1 = np.arange(-np.floor(f1.shape[0] / 2.), np.ceil(f1.shape[0] / 2.))
    xi2 = np.arange(-np.floor(f1.shape[1] / 2.), np.ceil(f1.shape[1] / 2.))
    XI1, XI2 = np.meshgrid(xi1, xi2, indexing='ij')
    phase = np.exp(-2 * np.pi * 1j * delta * (XI1 + XI2))
    f1psh = f1h * phase

    return f1, f1shift, f1h, f1sh, f1psh


def imrot(f, t):
    """
    Rotate array f around its center by t radians counterclockwise
    """
    nx = f.shape[0]
    ny = f.shape[1]
    p0 = nx / 2
    q0 = ny / 2

    #X, Y = np.meshgrid(np.arange(p0, p0 + nx) % nx,
    #                   np.arange(q0, q0 + ny) % ny,
    #                   indexing='ij')
    X, Y = np.meshgrid(np.arange(0, nx),
                       np.arange(0, ny),
                       indexing='ij')

    R = np.array([[np.cos(-t), -np.sin(-t)],
                  [np.sin(-t), np.cos(-t)]])
    C = np.c_[X[..., None], Y[..., None]] - np.array([p0, q0])[None, None, :]
    RC = np.einsum('ij,abj->abi', R, C) + np.array([p0, q0])[None, None, :]
    #Rfr = map_coordinates(f.real, RC.transpose(2, 0, 1), order=1, mode='wrap')
    #Rfi = map_coordinates(f.imag, RC.transpose(2, 0, 1), order=1, mode='wrap')
    Rfr = map_wrap(f.real, RC.transpose(2, 0, 1))
    Rfi = map_wrap(f.imag, RC.transpose(2, 0, 1))

    return Rfr + Rfi * 1j


def shift_fft(f):
    nx = f.shape[0]
    ny = f.shape[1]
    p0 = nx / 2
    q0 = ny / 2

    X, Y = np.meshgrid(np.arange(p0, p0 + nx) % nx,
                       np.arange(q0, q0 + ny) % ny,
                       indexing='ij')
    fs = f[X, Y, ...]

    return T2FFT.analyze(fs, axes=(0, 1))

def shift_ifft(fh):
    nx = fh.shape[0]
    ny = fh.shape[1]
    p0 = nx / 2
    q0 = ny / 2

    X, Y = np.meshgrid(np.arange(-p0, -p0 + nx) % nx,
                       np.arange(-q0, -q0 + ny) % ny,
                       indexing='ij')
    fs = T2FFT.synthesize(fh, axes=(0, 1))

    f = fs[X, Y, ...]

    return f

def test_ft_rotation3():

    nx = 20
    ny = 20
    nt = 10
    p0 = nx / 2
    q0 = ny / 2

    X, Y = np.meshgrid(np.arange(0, nx),
                       np.arange(0, ny),
                       indexing='ij')
    C = np.c_[X[..., None], Y[..., None]]

    F = SE2_FFT(spatial_grid_size=(nx, ny, nt),
                interpolation_method='spline',
                oversampling_factor=5,
                spline_order=1)

    fs = []
    f1cs = []
    f1ps = []
    f2s = []
    f2fs = []
    fhs = []
    for i in range(36):
        t = i * 2 * np.pi / 36.
        R = np.array([[np.cos(-t), -np.sin(-t)],
                      [np.sin(-t), np.cos(-t)]])
        RC = np.einsum('ij,abj->abi', R, C - np.array([p0, q0])[None, None, :]) + np.array([p0, q0])[None, None, :]
        #Rfr = map_wrap(f1.real, RC.transpose(2, 0, 1))
        #Rfi = map_wrap(f1.imag, RC.transpose(2, 0, 1))
        #Rf = Rfr + Rfi * 1j

        Rf = np.exp(1j * 2 * np.pi * (5 * RC[..., 0, None] * np.ones(nt)[None, None, :]) / nx)

        f, f1c, f1p, f2, f2f, f_hat = F.analyze(Rf)

        fs.append(f)
        f1cs.append(f1c)
        f1ps.append(f1p)
        f2s.append(f2)
        f2fs.append(f2f)
        fhs.append(f_hat)


    return fs, f1cs, f1ps, f2s, f2fs, fhs


def test_ft_rotation2(t=np.pi/10):

    nx = 20
    ny = 20
    p0 = nx / 2
    q0 = ny / 2

    # Shows that when the image rotates around center (p0, q0), the FT also rotates around (p0, q0) (which corresponds
    # to frequency (0, 0).
    #f1 = np.zeros((nx, ny), dtype='complex')
    #f1[p0 - 3:p0 + 4, q0 - 3:q0+4] = np.random.randn(7, 7) + 1j * np.random.randn(7, 7)

    X, Y = np.meshgrid(np.arange(0, nx),
                       np.arange(0, ny),
                       indexing='ij')
    C = np.c_[X[..., None], Y[..., None]]


    #f1 = np.exp(1j * 2 * np.pi * (5 * X) / nx)

    fs = []
    fhs = []

    for i in range(36):
        t = i * 2 * np.pi / 36.
        R = np.array([[np.cos(-t), -np.sin(-t)],
                      [np.sin(-t), np.cos(-t)]])
        RC = np.einsum('ij,abj->abi', R, C - np.array([p0, q0])[None, None, :]) + np.array([p0, q0])[None, None, :]
        #Rfr = map_wrap(f1.real, RC.transpose(2, 0, 1))
        #Rfi = map_wrap(f1.imag, RC.transpose(2, 0, 1))
        #Rf = Rfr + Rfi * 1j

        Rf = np.exp(1j * 2 * np.pi * (5 * RC[..., 0]) / nx)
        Rfh = shift_fft(Rf)
        Rfh = T2FFT.analyze(Rf)

        fs.append(Rf)
        fhs.append(Rfh)

    return fs, fhs


def test_ft_rotation(t=np.pi/3.):

    nx = 500
    ny = 500
    p0 = nx / 2
    q0 = ny / 2

    # Shows that when the image rotates around center (p0, q0), the FT also rotates around (p0, q0) (which corresponds
    # to frequency (0, 0).
    f1 = np.zeros((nx, ny), dtype='complex')
    #f1[p0 - 1, q0 - 1] = 1.
    #f1[p0, q0] = 1.
    #f1[p0 + 1, q0 + 1] = 1.
    #f1[p0 + 2, q0 + 2] = 1.
    f1[p0 - 3:p0 + 4, q0 - 3:q0+4] = np.random.randn(7, 7) + 1j * np.random.randn(7, 7)

    f2 = np.zeros((nx, ny))
    #f2[p0 - 1, q0 + 1] = 1.
    #f2[p0, q0] = 1.
    #f2[p0 + 1, q0 - 1] = 1.
    #f2[p0 + 2, q0 - 2] = 1.
    f2 = imrot(f1, t)

    #F = SE2_FFT(spatial_grid_size=(nx, ny, 10),
    #            interpolation_method='spline',
    #            spline_order=1,
    #            oversampling_factor=5)
    #f1hp = F.resample_c2p(f1hc)
    #f2hp = F.resample_c2p(f2hc)

    #f1 = f1[:, :, None] * np.ones(10)[None, None, :]
    #f2 = f2[:, :, None] * np.ones(10)[None, None, :]


    #f1, f11c, f11p, f12, f12f, f1_hat = F.analyze(f1)
    #f2, f21c, f21p, f22, f22f, f2_hat = F.analyze(f2)

    #f11c_irot = imrot(f11c[:, :, 0], -t)

    f1h = shift_fft(f1)
    f2h = shift_fft(f2)

    f1h_rot = imrot(f1h, t)
    f1h_irot = imrot(f1h, -t)

    return f1, f2, f1h, f2h, f1h_rot, f1h_irot


def cartesian_grid(nx, ny):

    x = np.linspace(-0.5, 0.5, nx, endpoint=False)
    y = np.linspace(-0.5, 0.5, ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')
    return X, Y


class SE2_FFT(FFTBase):

    def __init__(self,
                 spatial_grid_size=(10, 10, 10),
                 interpolation_method='spline',
                 spline_order=1,
                 oversampling_factor=1):

        self.spatial_grid_size = spatial_grid_size  # tau_x, tau_y, theta
        self.interpolation_method = interpolation_method

        if interpolation_method == 'spline':
            self.spline_order = spline_order

            # The array coordinates of the zero-frequency component
            self.p0 = spatial_grid_size[0] // 2
            self.q0 = spatial_grid_size[1] // 2

            # The distance, in pixels, from the (0, 0) pixel to the center of frequency space
            self.r_max = np.sqrt(self.p0 ** 2 + self.q0 ** 2)

            # Precomputation for cartesian-to-polar regridding
            self.n_samples_r = oversampling_factor * (np.ceil(self.r_max) + 1)
            self.n_samples_t = oversampling_factor * (np.ceil(2 * np.pi * self.r_max))

            r = np.linspace(0, self.r_max, self.n_samples_r, endpoint=True)
            theta = np.linspace(0, 2 * np.pi, self.n_samples_t, endpoint=False)
            R, THETA, = np.meshgrid(r, theta, indexing='ij')

            # Convert polar to Cartesian coordinates
            X = R * np.cos(THETA)
            Y = R * np.sin(THETA)

            # Transform to array indices (note; these are not the usual coordinates where y axis is flipped)
            I = X + self.p0
            J = Y + self.q0

            self.c2p_coords = np.r_[I[None, ...], J[None, ...]]


            # Precomputation for polar-to-cartesian regridding
            i = np.arange(0, self.spatial_grid_size[0])
            j = np.arange(0, self.spatial_grid_size[1])
            x = i - self.p0
            y = j - self.q0
            X, Y = np.meshgrid(x, y, indexing='ij')

            # Convert Cartesian to polar coordinates:
            R = np.sqrt(X ** 2 + Y ** 2)
            T = np.arctan2(Y, X) #  % (2 * np.pi)

            # Convert to array indices
            # Maximum of R is r_max, maximum index in array is (n_samples_r - 1)
            R *= (self.n_samples_r - 1) / self.r_max
            # The maximum angle in T is arbitrarily close to 2 pi,
            # but this should end up 1 pixel past the last index n_samples_t - 1, i.e. it should end up at n_samples_t
            # which is equal to index 0 since wraparound is used.
            T *= self.n_samples_t / (2 * np.pi)

            self.p2c_coords = np.r_[R[None, ...], T[None, ...]]
        elif interpolation_method == 'Fourier':

            #r_max = np.sqrt(2)
            r_max = 1. / np.sqrt(2.)
            #nr = spatial_grid_size[0] + 1
            nr = 15 * np.ceil(r_max * spatial_grid_size[0])
            nt = 5 * np.ceil(2 * np.pi * r_max * spatial_grid_size[0])
            nx = spatial_grid_size[0]
            ny = spatial_grid_size[1]
            self.flerp = FourierInterpolator.init_cartesian_to_polar(nr, nt, nx, ny)

        else:
            raise ValueError('Unknown interpolation method:' + str(interpolation_method))

    def analyze(self, f):
        """
        Compute the SE(2) Fourier Transform of a function f : SE(2) -> C or f : SE(2) -> R.
        The SE(2) Fourier Transform expands f in the basis of matrix elements of irreducible representations of SE(2).
        Let T^r_pq(g) be the (p, q) matrix element of the irreducible representation of SE(2) of weight / radius r,
        then the FT is:

        F^r_pq = int_SE(2) f(g) conjugate(T^r_pq(g^{-1})) dg

        We assume g in SE(2) to be parameterized as g = (tau_x, tau_y, theta), where tau is a 2D translation vector
        and theta is a rotation angle.
        The input f is a 3D array of shape (N_x, N_y, N_t),
        where the axes correspond to tau_x, tau_y, theta in the ranges:
        tau_x in np.linspace(-0.5, 0.5, N_x, endpoint=False)
        tau_y in np.linspace(-0.5, 0.5, N_y, endpoint=False)
        theta in np.linspace(0, 2 * np.pi, N_t, endpoint=False)

        See:
        "Engineering Applications of Noncommutative Harmonic Analysis", section 11.2
        Chrikjian & Kyatkin

        "The Mackey Machine: a user manual"
        Taco S. Cohen

        :param f: discretely sampled function on SE(2).
         The first two axes of f correspond to translation parameters tau_x, tau_y, and the third axis corresponds to
         rotation angle theta.
        :return: F, the SE(2) Fourier Transform of f. Axes of F are (r, p, q)
        """

        # First, FFT along translation parameters tau_1 and tau_2
        #f1c_shift = T2FFT.analyze(f, axes=(0, 1))
        # This gives: f1c_shift[xi_1, xi_2, theta]
        # where xi_1 and xi_2 are Cartesian (c) coordinates of the frequency domain.
        # However, this is the FT of the *shifted* function on [0, 1), so shift the coefficient back:
        #delta = -0.5  # we're shifting from [0, 1) to [-0.5, 0.5)
        #xi1 = np.arange(-np.floor(f1c_shift.shape[0] / 2.), np.ceil(f1c_shift.shape[0] / 2.))
        #xi2 = np.arange(-np.floor(f1c_shift.shape[1] / 2.), np.ceil(f1c_shift.shape[1] / 2.))
        #XI1, XI2 = np.meshgrid(xi1, xi2, indexing='ij')
        #phase = np.exp(-2 * np.pi * 1j * delta * (XI1 + XI2))
        #f1c = f1c_shift * phase[:, :, None]

        f1c = shift_fft(f)

        # Change from Cartesian (c) to a polar (p) grid:
        f1p = self.resample_c2p_3d(f1c)
        # This gives f1p[r, varphi, theta]

        # FFT along rotation angle theta
        # We conjugate the argument and the ouput so that the complex exponential has positive instead of negative sign
        f2 = T1FFT.analyze(f1p.conj(), axis=2).conj()
        # This gives f2[r, varphi, q]
        # where q ranges from q = -floor(f1p.shape[2] / 2) to q = ceil(f1p.shape[2] / 2) - 1  (inclusive)

        # Multiply f2 by a (varphi, q)-dependent phase factor:
        m_min = -np.floor(f2.shape[2] / 2.)
        m_max = np.ceil(f1p.shape[2] / 2.) - 1
        varphi = np.linspace(0, 2 * np.pi, f2.shape[1], endpoint=False)  # may not need this many points on every circle
        factor = np.exp(-1j * varphi[None, :, None] * np.arange(m_min, m_max + 1)[None, None, :])
        f2f = f2 * factor

        # FFT along polar coordinate of frequency domain
        f_hat = T1FFT.analyze(f2f.conj(), axis=1).conj()
        # This gives f_hat[r, p, q]

        return f, f1c, f1p, f2, f2f, f_hat

    def synthesize(self, f_hat):

        f2f = T1FFT.synthesize(f_hat.conj(), axis=1).conj()

        # Multiply f_2 by a phase factor:
        m_min = -np.floor(f2f.shape[2] / 2)
        m_max = np.ceil(f2f.shape[2] / 2) - 1
        psi = np.linspace(0, 2 * np.pi, f2f.shape[1], endpoint=False)  # may not need this many points on every circle
        factor = np.exp(1j * psi[:, None] * np.arange(m_min, m_max + 1)[None, :])

        f2 = f2f * factor[None, ...]

        f1p = T1FFT.synthesize(f2.conj(), axis=2).conj()

        f1c = self.resample_p2c_3d(f1p)


        # delta = -0.5  # we're shifting from [0, 1) to [-0.5, 0.5)
        # xi1 = np.arange(-np.floor(f1c.shape[0] / 2), np.ceil(f1c.shape[0] / 2))
        # xi2 = np.arange(-np.floor(f1c.shape[1] / 2), np.ceil(f1c.shape[1] / 2))
        # XI1, XI2 = np.meshgrid(xi1, xi2, indexing='ij')
        # phase = np.exp(-2 * np.pi * 1j * delta * (XI1 + XI2))
        # f1c_shift = f1c / phase[:, :, None]


        #f = T2FFT.synthesize(f1c, axes=(0, 1))
        #f = T2FFT.synthesize(f1c_shift, axes=(0, 1))
        f = shift_ifft(f1c)


        return f, f1c, f1p, f2, f2f, f_hat

    def resample_c2p(self, fc):
        """
        Resample a function on a Cartesian grid to a polar grid.

        The center of the Cartesian coordinate system is assumed to be in the center of the image at index
        x0 = fc.shape[0] / 2 - 0.5
        y0 = fc.shape[1] / 2 - 0.5
        i.e. for a 2-pixel image, x0 would be at 'index' 2/2-0.5 = 0.5, in between the two pixels.

        The first dimension of the output coresponds to the radius r in [0, r_max=fc.shape[0] / 2. - 0.5]
        while the second dimension corresponds to the angle theta in [0, 2pi).

        :param fc: function values sampled on a Cartesian grid.
        :return: resampled function on a polar grid
        """

        # We are dealing with three coordinate frames:
        # The array indices / image coordinates (i, j) of the input data array.
        # The Cartesian frame (x, y) centered in the image, with the same directions and units (=pixels) on the axes.
        # The polar coordinate frame (r, theta), also centered in the image, with theta=0 corresponding to the x axis.

        # (x0, y0) are the image coordinates / array indices of the center of the Cartesian coordinate frame
        # centered in the image. Note that although they are in the image coordinate frame, they are not necessarily ints.
        #fp_r = map_coordinates(fc.real, self.c2p_coords, order=self.spline_order, mode='wrap')  # 'nearest')
        #fp_c = map_coordinates(fc.imag, self.c2p_coords, order=self.spline_order, mode='wrap')  # 'nearest')
        #fp = fp_r + 1j * fp_c
        fp_r = map_wrap(fc.real, self.c2p_coords)
        fp_c = map_wrap(fc.imag, self.c2p_coords)
        fp = fp_r + 1j * fp_c

        return fp

    def resample_p2c(self, fp): # , order=1, mode='wrap', cval=np.nan):

        fc_r = map_coordinates(fp.real, self.p2c_coords, order=self.spline_order, mode='wrap')
        fc_c = map_coordinates(fp.imag, self.p2c_coords, order=self.spline_order, mode='wrap')
        fc = fc_r + 1j * fc_c
        return fc

    def resample_c2p_3d(self, fc):

        if self.interpolation_method == 'spline':
            fp = []
            for i in range(fc.shape[2]):
                fp.append(self.resample_c2p(fc[:, :, i]))

            return np.c_[fp].transpose(1, 2, 0)

        elif self.interpolation_method == 'Fourier':

            fp = []
            for i in range(fc.shape[2]):
                fp.append(self.flerp.forward(fc[:, :, i]))

            return np.c_[fp].transpose(1, 2, 0)


    def resample_p2c_3d(self, fp):

        if self.interpolation_method == 'spline':
            fc = []
            for i in range(fp.shape[2]):
                fc.append(self.resample_p2c(fp[:, :, i]))

            return np.c_[fc].transpose(1, 2, 0)

        elif self.interpolation_method == 'Fourier':
            fc = []
            for i in range(fp.shape[2]):
                fc.append(self.flerp.backward(fp[:, :, i]))

            return np.c_[fc].transpose(1, 2, 0)


def R2_SE2_FFT(f):
    """
    Compute the SE(2) Fourier Transform of f : R^2 -> R as if f is a function on SE(2).
    That is, we view f as a function on SE(2):
    f+(g) = (L_g f)(0) = f(g^{-1} 0)

    If we parameterize g = (r, theta), where r is a translation vector and theta a rotation angle,
    we see that f+ is constant along theta, because a rotation of 0 is 0.
    Therefore,
    """
    pass

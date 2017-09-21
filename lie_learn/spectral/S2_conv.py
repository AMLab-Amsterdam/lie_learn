
import numpy as np

import lie_learn.spaces.S2 as S2
import lie_learn.groups.SO3 as SO3
from lie_learn.spectral.S2FFT import S2_FT_Naive
from lie_learn.spectral.SO3FFT_Naive import SO3_FT_Naive


def conv_test():
    """

    :return:
    """
    from lie_learn.spectral.SO3FFT_Naive import SO3_FT_Naive

    b = 10
    f1 = np.ones((2 * b + 2, b + 1))
    f2 = np.ones((2 * b + 2, b + 1))

    s2_fft = S2_FT_Naive(L_max=b - 1, grid_type='Gauss-Legendre',
                         field='real', normalization='quantum', condon_shortley='cs')

    so3_fft = SO3_FT_Naive(L_max=b - 1,
                           field='real', normalization='quantum', order='centered', condon_shortley='cs')

    # Spherical Fourier transform
    f1_hat = s2_fft.analyze(f1)
    f2_hat = s2_fft.analyze(f2)

    # Perform block-wise outer product
    f12_hat = []
    for l in range(b):
        f1_hat_l = f1_hat[l ** 2:l ** 2 + 2 * l + 1]
        f2_hat_l = f2_hat[l ** 2:l ** 2 + 2 * l + 1]

        f12_hat_l = f1_hat_l[:, None] * f2_hat_l[None, :].conj()
        f12_hat.append(f12_hat_l)

    # Inverse SO(3) Fourier transform
    f12 = so3_fft.synthesize(f12_hat)

    return f12


def spectral_S2_conv(f1, f2, s2_fft=None, so3_fft=None):
    """
    Compute the convolution of two functions on the 2-sphere.
    Let f1 : S^2 -> R and f2 : S^2 -> R, then the convolution is defined as
    f1 * f2(g) = int_{S^2} f1(x) f2(g^{-1} x) dx,
    where g in SO(3) and dx is the normalized Haar measure on S^2.

    The convolution is computed by a Fourier transform.
    It can be shown that the SO(3)-Fourier transform of the convolution f1 * f2 is equal to the outer product
    of the spherical Fourier transform of f1 and f2.
    Specifically, let f1_hat be the spherical FT of f1 and f2_hat the spherical FT of f2.
    These vectors are split into chunks of dimension 2l+1, for l=0, ..., b (the bandwidth)
    For each degree, we take the outer product to obtain a (2l+1) x (2l+1) matrix, which is the degree-l
    block of the FT of f1*f2.
    For more details, see our note on "Convolution on S^2 and SO(3)"

    :param f1:
    :param f2:
    :param s2_fft:
    :param so3_fft:
    :return:
    """

    b = f1.shape[1] - 1  # TODO we assume a Gauss-Legendre grid for S^2 here

    if s2_fft is None:
        s2_fft = S2_FT_Naive(L_max=b - 1, grid_type='Gauss-Legendre',
                             field='real', normalization='quantum', condon_shortley='cs')

    if so3_fft is None:
        so3_fft = SO3_FT_Naive(L_max=b - 1,
                               field='real', normalization='quantum', order='centered', condon_shortley='cs')

    # Spherical Fourier transform
    f1_hat = s2_fft.analyze(f1)
    f2_hat = s2_fft.analyze(f2)

    # Perform block-wise outer product
    f12_hat = []
    for l in range(b):
        f1_hat_l = f1_hat[l ** 2:l ** 2 + 2 * l + 1]
        f2_hat_l = f2_hat[l ** 2:l ** 2 + 2 * l + 1]

        f12_hat_l = f1_hat_l[:, None] * f2_hat_l[None, :].conj()
        f12_hat.append(f12_hat_l)

    # Inverse SO(3) Fourier transform
    return so3_fft.synthesize(f12_hat)


def naive_S2_conv(f1, f2, alpha, beta, gamma, g_parameterization='EA323'):
    """
    Compute int_S^2 f1(x) f2(g^{-1} x)* dx,
    where x = (theta, phi) is a point on the sphere S^2,
    and g = (alpha, beta, gamma) is a point in SO(3) in Euler angle parameterization

    :param f1, f2: functions to be convolved
    :param alpha, beta, gamma: the rotation at which to evaluate the result of convolution
    :return:
    """
    # This fails
    def integrand(theta, phi):
        g_inv = SO3.invert((alpha, beta, gamma), parameterization=g_parameterization)
        g_inv_theta, g_inv_phi, _ = SO3.transform_r3(g=g_inv, x=(theta, phi, 1.),
                                                     g_parameterization=g_parameterization, x_parameterization='S')
        return f1(theta, phi) * f2(g_inv_theta, g_inv_phi).conj()

    return S2.integrate(f=integrand, normalize=True)


def naive_S2_conv_v2(f1, f2, alpha, beta, gamma, g_parameterization='EA323'):
    """
    Compute int_S^2 f1(x) f2(g^{-1} x)* dx,
    where x = (theta, phi) is a point on the sphere S^2,
    and g = (alpha, beta, gamma) is a point in SO(3) in Euler angle parameterization

    :param f1, f2: functions to be convolved
    :param alpha, beta, gamma: the rotation at which to evaluate the result of convolution
    :return:
    """

    theta, phi = S2.meshgrid(b=3, grid_type='Gauss-Legendre')
    w = S2.quadrature_weights(b=3, grid_type='Gauss-Legendre')

    print(theta.shape, phi.shape)
    s2_coords = np.c_[theta[..., None], phi[..., None]]
    print(s2_coords.shape)
    r3_coords = np.c_[theta[..., None], phi[..., None], np.ones_like(theta)[..., None]]

    # g_inv = SO3.invert((alpha, beta, gamma), parameterization=g_parameterization)
    # g_inv = (-gamma, -beta, -alpha)
    g_inv = (alpha, beta, gamma)  # wrong

    ginvx = SO3.transform_r3(g=g_inv, x=r3_coords, g_parameterization=g_parameterization, x_parameterization='S')
    print(ginvx.shape)
    g_inv_theta = ginvx[..., 0]
    g_inv_phi = ginvx[..., 1]
    g_inv_r = ginvx[..., 2]

    print(g_inv_theta, g_inv_phi, g_inv_r)

    f1_grid = f1(theta, phi)
    f2_grid = f2(g_inv_theta, g_inv_phi)

    print(f1_grid.shape, f2_grid.shape, w.shape)
    return np.sum(f1_grid * f2_grid * w)




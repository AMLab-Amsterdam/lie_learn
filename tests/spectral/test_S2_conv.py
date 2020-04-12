
import numpy as np

import lie_learn.spaces.S2 as S2
import lie_learn.spaces.S3 as S3
import lie_learn.groups.SO3 as SO3
from lie_learn.representations.SO3.spherical_harmonics import sh
from lie_learn.spectral.S2_conv import naive_S2_conv, spectral_S2_conv, naive_S2_conv_v2


def compare_naive_and_spectral_conv():

    f1 = lambda t, p: sh(l=2, m=1, theta=t, phi=p, field='real', normalization='quantum', condon_shortley=True)
    f2 = lambda t, p: sh(l=2, m=1, theta=t, phi=p, field='real', normalization='quantum', condon_shortley=True)

    theta, phi = S2.meshgrid(b=4, grid_type='Gauss-Legendre')
    f1_grid = f1(theta, phi)
    f2_grid = f2(theta, phi)

    alpha, beta, gamma = S3.meshgrid(b=4, grid_type='SOFT')  # TODO check convention

    f12_grid_spectral = spectral_S2_conv(f1_grid, f2_grid, s2_fft=None, so3_fft=None)

    f12_grid = np.zeros_like(alpha)
    for i in range(alpha.shape[0]):
        for j in range(alpha.shape[1]):
            for k in range(alpha.shape[2]):
                f12_grid[i, j, k] = naive_S2_conv(f1, f2, alpha[i, j, k], beta[i, j, k], gamma[i, j, k])
                print(i, j, k, f12_grid[i, j, k])

    return f1_grid, f2_grid, f12_grid, f12_grid_spectral


def naive_conv(l1=1, m1=1, l2=1, m2=1, g_parameterization='EA313'):
    f1 = lambda t, p: sh(l=l1, m=m1, theta=t, phi=p, field='real', normalization='quantum', condon_shortley=True)
    f2 = lambda t, p: sh(l=l2, m=m2, theta=t, phi=p, field='real', normalization='quantum', condon_shortley=True)

    theta, phi = S2.meshgrid(b=3, grid_type='Gauss-Legendre')
    f1_grid = f1(theta, phi)
    f2_grid = f2(theta, phi)

    alpha, beta, gamma = S3.meshgrid(b=3, grid_type='SOFT')  # TODO check convention

    f12_grid = np.zeros_like(alpha)
    for i in range(alpha.shape[0]):
        for j in range(alpha.shape[1]):
            for k in range(alpha.shape[2]):
                f12_grid[i, j, k] = naive_S2_conv_v2(f1, f2, alpha[i, j, k], beta[i, j, k], gamma[i, j, k], g_parameterization)
                print(i, j, k, f12_grid[i, j, k])

    return f1_grid, f2_grid, f12_grid

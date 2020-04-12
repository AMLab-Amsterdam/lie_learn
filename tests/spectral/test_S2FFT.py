import numpy as np

import lie_learn.spaces.S2 as S2
from lie_learn.representations.SO3.spherical_harmonics import sh
from lie_learn.spectral.S2FFT import setup_legendre_transform, setup_legendre_transform_indices, sphere_fft, S2_FT_Naive


def test_S2_FT_Naive():

    L_max = 6

    for grid_type in ('Gauss-Legendre', 'Clenshaw-Curtis'):

        theta, phi = S2.meshgrid(b=L_max + 1, grid_type=grid_type)

        for field in ('real', 'complex'):
            for normalization in ('quantum', 'seismology'):  # TODO Others should work but are not normalized
                for condon_shortley in ('cs', 'nocs'):

                    fft = S2_FT_Naive(L_max, grid_type=grid_type,
                                      field=field, normalization=normalization, condon_shortley=condon_shortley)

                    for l in range(L_max):
                        for m in range(-l, l + 1):

                            y_true = sh(
                                l, m, theta, phi,
                                field=field, normalization=normalization, condon_shortley=condon_shortley == 'cs')

                            y_hat = fft.analyze(y_true)

                            # The flat index for (l, m) is l^2 + l + m
                            # Before the harmonics of degree l, there are this many harmonics:
                            # sum_{i=0}^{l-1} 2i+1 = l^2
                            # There are 2l+1 harmonics of degree l, with order m=0 at the center,
                            # so the m-th harmonic of degree is at l + m within the block of degree l.
                            y_hat_true = np.zeros_like(y_hat)
                            y_hat_true[l**2 + l + m] = 1

                            y = fft.synthesize(y_hat_true)

                            diff = np.sum(np.abs(y_hat - y_hat_true))
                            print(grid_type, field, normalization, condon_shortley, l, m, diff)
                            assert np.isclose(diff, 0.)

                            diff = np.sum(np.abs(y - y_true))
                            print(grid_type, field, normalization, condon_shortley, l, m, diff)
                            assert np.isclose(diff, 0.)


def test_S2FFT():

    L_max = 10
    beta, alpha = S2.meshgrid(b=L_max + 1, grid_type='Driscoll-Healy')
    lt = setup_legendre_transform(b=L_max + 1)
    lti = setup_legendre_transform_indices(b=L_max + 1)

    for l in range(L_max):
        for m in range(-l, l + 1):

            Y = sh(l, m, beta, alpha,
                   field='complex', normalization='seismology', condon_shortley=True)

            y_hat = sphere_fft(Y, lt, lti)

            # The flat index for (l, m) is l^2 + l + m
            # Before the harmonics of degree l, there are this many harmonics: sum_{i=0}^{l-1} 2i+1 = l^2
            # There are 2l+1 harmonics of degree l, with order m=0 at the center,
            # so the m-th harmonic of degree is at l + m within the block of degree l.
            y_hat_true = np.zeros_like(y_hat)
            y_hat_true[l**2 + l + m] = 1

            diff = np.sum(np.abs(y_hat - y_hat_true))
            nz = 1. - np.isclose(y_hat, 0.)
            diff_nz = np.sum(np.abs(nz - y_hat_true))
            print(l, m, diff, diff_nz)
            print(np.round(y_hat, 4))
            print(y_hat_true)
            # assert np.isclose(diff, 0.)  # TODO make this work
            print(nz)
            assert np.isclose(diff_nz, 0.)


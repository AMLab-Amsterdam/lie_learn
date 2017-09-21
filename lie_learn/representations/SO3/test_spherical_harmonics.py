import numpy as np
import lie_learn.spaces.S2 as S2
from lie_learn.representations.SO3.spherical_harmonics import sh, sh_squared_norm


def check_orthogonality(L_max=3, grid_type='Gauss-Legendre',
                        field='real', normalization='quantum', condon_shortley=True):

    theta, phi = S2.meshgrid(b=L_max + 1, grid_type=grid_type)
    w = S2.quadrature_weights(b=L_max + 1, grid_type=grid_type)

    for l in range(L_max):
        for m in range(-l, l + 1):
            for l2 in range(L_max):
                for m2 in range(-l2, l2 + 1):
                    Ylm = sh(l, m, theta, phi, field, normalization, condon_shortley)
                    Ylm2 = sh(l2, m2, theta, phi, field, normalization, condon_shortley)

                    dot_numerical = S2.integrate_quad(Ylm * Ylm2.conj(), grid_type=grid_type, normalize=False, w=w)

                    dot_numerical2 = S2.integrate(
                        lambda t, p: sh(l, m, t, p, field, normalization, condon_shortley) * \
                                     sh(l2, m2, t, p, field, normalization, condon_shortley).conj(), normalize=False)

                    sqnorm_analytical = sh_squared_norm(l, normalization, normalized_haar=False)
                    dot_analytical = sqnorm_analytical * (l == l2 and m == m2)

                    print(l, m, l2, m2, field, normalization, condon_shortley, dot_analytical, dot_numerical, dot_numerical2)
                    assert np.isclose(dot_numerical, dot_analytical)
                    assert np.isclose(dot_numerical2, dot_analytical)


def test_orthogonality():
    L_max = 2
    grid_type = 'Gauss-Legendre'

    for field in ('real', 'complex'):
        for normalization in ('quantum', 'seismology', 'geodesy', 'nfft'):
            for condon_shortley in (True, False):
                check_orthogonality(L_max, grid_type, field, normalization, condon_shortley)

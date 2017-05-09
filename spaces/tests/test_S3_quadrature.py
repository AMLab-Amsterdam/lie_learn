import numpy as np

import lie_learn.spaces.S3 as S3
from lie_learn.representations.SO3.wigner_d import wigner_D_function


def test_S3_quadint_equals_numint():
    """Test if SO(3) quadrature integration gives the same result as scipy numerical integration"""
    b = 10
    for l in range(2):
        for m in range(-l, l + 1):
            for n in range(-l, l + 1):
                check_S3_quadint_equals_numint(l, m, n, b)


def check_S3_quadint_equals_numint(l=1, m=1, n=1, b=10):
    # Create grids on the sphere
    x = S3.meshgrid(b=b, grid_type='SOFT')
    x = np.c_[x[0][..., None], x[1][..., None], x[2][..., None]]

    # Compute quadrature weights
    w = S3.quadrature_weights(b=b, grid_type='SOFT')

    # Define a polynomial function, to be evaluated at one point or at an array of points
    def f1(alpha, beta, gamma):
        df = wigner_D_function(l=l, m=m, n=n, alpha=alpha, beta=beta, gamma=gamma)
        return df * df.conj()

    def f1a(xs):
        d = np.zeros(x.shape[:-1])
        for i in range(d.shape[0]):
            for j in range(d.shape[1]):
                for k in range(d.shape[2]):
                    d[i, j, k] = f1(xs[i, j, k, 0], xs[i, j, k, 1], xs[i, j, k, 2])
        return d

    # Obtain the "true" value of the integral of the function over the sphere, using scipy's numerical integration
    # routines
    i1 = S3.integrate(f1, normalize=True)

    # Compute the integral using the quadrature formulae
    i1_w = S3.integrate_quad(f1a(x), grid_type='SOFT', normalize=True, w=w)

    # Check error
    print(b, l, m, n, 'results:', i1_w, i1, 'diff:', np.abs(i1_w - i1))
    assert np.isclose(np.abs(i1_w - i1), 0.0)


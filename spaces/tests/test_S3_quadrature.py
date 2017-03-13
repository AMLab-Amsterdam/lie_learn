
import lie_learn.spaces.S3 as S3
import numpy as np


def test_S3_quadrature():
    """
    Testing spherical quadrature rule versus numerical integration.
    """

    b = 10

    # Create grids on the sphere
    x = S3.meshgrid(b=b)
    x = np.c_[x[0][..., None], x[1][..., None], x[2][..., None]]

    # Compute quadrature weights
    w = S3.quadrature_weights(b=b)

    # Define a polynomial function, to be evaluated at one point or at an array of points
    def f1a(xs):
        xc = S3.change_coordinates(coords=xs, p_from='S', p_to='C')
        return (xc[..., 0] ** 2 * xc[..., 1] - 1.4 * xc[..., 2] * xc[..., 1] ** 3 + xc[..., 1] - xc[..., 2] ** 2 + 2.
                + 2.3 * xc[..., 3] ** 2 * xc[..., 1] + 0.7 * xc[..., 3])

    def f1(alpha, beta, gamma):
        xs = np.array([alpha, beta, gamma])
        return f1a(xs)

    # Obtain the "true" value of the integral of the function over the sphere, using scipy's numerical integration
    # routines
    i1 = S3.integrate(f1, normalize=False)

    # Compute the integral using the quadrature formulae
    i1_w = (w * f1a(x)).sum()
    print(i1_w, i1, 'diff:', np.abs(i1_w - i1))
    assert np.isclose(np.abs(i1_w - i1), 0.0)


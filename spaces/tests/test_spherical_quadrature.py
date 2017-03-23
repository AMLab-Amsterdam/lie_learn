
import lie_learn.spaces.S2 as S2
import numpy as np


def test_spherical_quadrature():
    """
    Testing spherical quadrature rule versus numerical integration.
    """

    b = 8  # 10

    # Create grids on the sphere
    x_gl = S2.meshgrid(b=b, grid_type='Gauss-Legendre')
    x_cc = S2.meshgrid(b=b, grid_type='Clenshaw-Curtis')
    x_soft = S2.meshgrid(b=b, grid_type='SOFT')
    x_gl = np.c_[x_gl[0][..., None], x_gl[1][..., None]]
    x_cc = np.c_[x_cc[0][..., None], x_cc[1][..., None]]
    x_soft = np.c_[x_soft[0][..., None], x_soft[1][..., None]]

    # Compute quadrature weights
    w_gl = S2.quadrature_weights(b=b, grid_type='Gauss-Legendre')
    w_cc = S2.quadrature_weights(b=b, grid_type='Clenshaw-Curtis')
    w_soft = S2.quadrature_weights(b=b, grid_type='SOFT')

    # Define a polynomial function, to be evaluated at one point or at an array of points
    def f1a(xs):
        xc = S2.change_coordinates(coords=xs, p_from='S', p_to='C')
        return xc[..., 0] ** 2 * xc[..., 1] - 1.4 * xc[..., 2] * xc[..., 1] ** 3 + xc[..., 1] - xc[..., 2] ** 2 + 2.
    def f1(theta, phi):
        xs = np.array([theta, phi])
        return f1a(xs)

    # Obtain the "true" value of the integral of the function over the sphere, using scipy's numerical integration
    # routines
    i1 = S2.integrate(f1, normalize=False)

    # Compute the integral using the quadrature formulae
    # i1_gl_w = (w_gl * f1a(x_gl)).sum()
    i1_gl_w = S2.integrate_quad(f1a(x_gl), grid_type='Gauss-Legendre', normalize=False, w=w_gl)
    print(i1_gl_w, i1, 'diff:', np.abs(i1_gl_w - i1))
    assert np.isclose(np.abs(i1_gl_w - i1), 0.0)

    # i1_cc_w = (w_cc * f1a(x_cc)).sum()
    i1_cc_w = S2.integrate_quad(f1a(x_cc), grid_type='Clenshaw-Curtis', normalize=False, w=w_cc)
    print(i1_cc_w, i1, 'diff:', np.abs(i1_cc_w - i1))
    assert np.isclose(np.abs(i1_cc_w - i1), 0.0)

    i1_soft_w = (w_soft * f1a(x_soft)).sum()
    print(i1_soft_w, i1, 'diff:', np.abs(i1_soft_w - i1))
    print(i1_soft_w)
    print(i1)
    # assert np.isclose(np.abs(i1_cc_w - i1), 0.0)  # TODO

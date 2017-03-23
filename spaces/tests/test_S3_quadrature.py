
import lie_learn.spaces.S3 as S3
import numpy as np


# TODO this is not working for non-trivial polynomials.
# The quadrature weights seem to work correctly for wigner D functions though

def test_S3_quadrature():
    """
    Testing spherical quadrature rule versus numerical integration.
    """

    b = 100

    # Create grids on the sphere
    x = S3.meshgrid(b=b, grid_type='SOFT')
    x = np.c_[x[0][..., None], x[1][..., None], x[2][..., None]]

    # Compute quadrature weights
    w = S3.quadrature_weights(b=b, grid_type='SOFT')

    # Define a polynomial function, to be evaluated at one point or at an array of points
    def f1a(xs):
        xc = S3.change_coordinates(coords=xs, p_from='S', p_to='C')
        return xc[..., 0] ** 0
        #return (xc[..., 0] ** 2 * xc[..., 1] - 1.4 * xc[..., 2] * xc[..., 1] ** 3 + xc[..., 1] - xc[..., 2] ** 2 + 2.
        #        + 2.3 * xc[..., 3] ** 2 * xc[..., 1] + 0.7 * xc[..., 3])

        print(xs.shape)
        d = np.zeros(x.shape[:-1])
        from lie_learn.representations.SO3.wigner_d import wigner_D_function
        for i in range(d.shape[0]):
            for j in range(d.shape[1]):
                for k in range(d.shape[2]):
                    d[i, j, k] = wigner_D_function(l=2, m=1, n=-1, alpha=xs[i, j, k, 0], beta=xs[i, j, k, 1], gamma=xs[i, j, k, 2])

    def f1(alpha, beta, gamma):
        xs = np.array([alpha, beta, gamma])
        return f1a(xs)

    # Obtain the "true" value of the integral of the function over the sphere, using scipy's numerical integration
    # routines
    i1 = S3.integrate(f1, normalize=True)

    # Compute the integral using the quadrature formulae
    # i1_w = (w * f1a(x)).sum()
    i1_w = S3.integrate_quad(f1a(x), grid_type='SOFT', normalize=True, w=w)
    print(i1_w, i1, 'diff:', np.abs(i1_w - i1))
    assert np.isclose(np.abs(i1_w - i1), 0.0)


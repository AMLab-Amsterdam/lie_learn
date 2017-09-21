from functools import lru_cache

import numpy as np
import lie_learn.spaces.S2 as S2


def change_coordinates(coords, p_from='C', p_to='S'):
    """
    Change Spherical to Cartesian coordinates and vice versa, for points x in S^3.

    We use the following coordinate system:
    https://en.wikipedia.org/wiki/N-sphere#Spherical_coordinates
    Except that we use the order (alpha, beta, gamma), where beta ranges from 0 to pi while alpha and gamma range from
    0 to 2 pi.

    x0 = r * cos(alpha)
    x1 = r * sin(alpha) * cos(gamma)
    x2 = r * sin(alpha) * sin(gamma) * cos(beta)
    x3 = r * sin(alpha * sin(gamma) * sin(beta)

    :param conversion:
    :param coords:
    :return:
    """
    if p_from == p_to:
        return coords
    elif p_from == 'S' and p_to == 'C':

        alpha = coords[..., 0]
        beta = coords[..., 1]
        gamma = coords[..., 2]
        r = 1.

        out = np.empty(alpha.shape + (4,))

        ca = np.cos(alpha)
        cb = np.cos(beta)
        cc = np.cos(gamma)
        sa = np.sin(alpha)
        sb = np.sin(beta)
        sc = np.sin(gamma)
        out[..., 0] = r * ca
        out[..., 1] = r * sa * cc
        out[..., 2] = r * sa * sc * cb
        out[..., 3] = r * sa * sc * sb
        return out

    elif p_from == 'C' and p_to == 'S':

        raise NotImplementedError
        x = coords[..., 0]
        y = coords[..., 1]
        z = coords[..., 2]
        w = coords[..., 3]
        r = np.sqrt((coords ** 2).sum(axis=-1))

        out = np.empty(x.shape + (3,))
        out[..., 0] = np.arccos(z)         # alpha
        out[..., 1] = np.arctan2(y, x)     # beta
        out[..., 2] = np.arctan2(y, x)     # gamma
        return out

    else:
        raise ValueError('Unknown conversion:' + str(p_from) + ' to ' + str(p_to))


def linspace(b, grid_type='SOFT'):
    """
    Compute a linspace on the 3-sphere.

    Since S3 is ismorphic to SO(3), we use the grid grid_type from:
    FFTs on the Rotation Group
    Peter J. Kostelec and Daniel N. Rockmore
    http://www.cs.dartmouth.edu/~geelong/soft/03-11-060.pdf
    :param b:
    :return:
    """
    # alpha = 2 * np.pi * np.arange(2 * b) / (2. * b)
    # beta = np.pi * (2 * np.arange(2 * b) + 1) / (4. * b)
    # gamma = 2 * np.pi * np.arange(2 * b) / (2. * b)

    beta, alpha = S2.linspace(b, grid_type)

    # According to this paper:
    # "Sampling sets and quadrature formulae on the rotation group"
    # We can just tack a sampling grid for S^1 to a sampling grid for S^2 to get a sampling grid for SO(3).
    gamma = 2 * np.pi * np.arange(2 * b) / (2. * b)

    return alpha, beta, gamma


def meshgrid(b, grid_type='SOFT'):
    return np.meshgrid(*linspace(b, grid_type), indexing='ij')


def integrate(f, normalize=True):
    """
    Integrate a function f : S^3 -> R over the 3-sphere S^3, using the invariant integration measure
    mu((alpha, beta, gamma)) = dalpha sin(beta) dbeta dgamma
    i.e. this returns
    int_S^3 f(x) dmu(x) = int_0^2pi int_0^pi int_0^2pi f(alpha, beta, gamma) dalpha sin(beta) dbeta dgamma

    :param f: a function of three scalar variables returning a scalar.
    :param normalize: if we use the measure dalpha sin(beta) dbeta dgamma,
      the integral of f(a,b,c)=1 over the 3-sphere gives 8 pi^2.
      If normalize=True, we divide the result of integration by this normalization constant, so that f integrates to 1.
      In other words, use the normalized Haar measure.
    :return: the integral of f over the 3-sphere
    """
    from scipy.integrate import quad

    f2 = lambda alpha, gamma: quad(lambda beta: f(alpha, beta, gamma) * np.sin(beta),
                                   a=0,
                                   b=np.pi)[0]
    f3 = lambda alpha: quad(lambda gamma: f2(alpha, gamma),
                            a=0,
                            b=2 * np.pi)[0]

    integral = quad(f3, 0, 2 * np.pi)[0]

    if normalize:
        return integral / (8 * np.pi ** 2)
    else:
        return integral


def integrate_quad(f, grid_type, normalize=True, w=None):
    """
    Integrate a function f : SO(3) -> R, sampled on a grid of type grid_type, using quadrature weights w.

    :param f: an ndarray containing function values on a grid
    :param grid_type: the type of grid used to sample f
    :param normalize: whether to use the normalized Haar measure or not
    :param w: the quadrature weights. If not given, they are computed.
    :return: the integral of f over S^2.
    """

    if grid_type == 'SOFT':
        b = f.shape[0] // 2

        if w is None:
            w = quadrature_weights(b, grid_type)

        integral = np.sum(f * w[None, :, None])
    else:
        raise NotImplementedError('Unsupported grid_type:', grid_type)

    if normalize:
        return integral
    else:
        return integral * 8 * np.pi ** 2


@lru_cache(maxsize=32)
def quadrature_weights(b, grid_type='SOFT'):
    """
    Compute quadrature weights for the grid used by Kostelec & Rockmore [1, 2].

    This grid is:
    alpha = 2 pi i / 2b
    beta = pi (2 j + 1) / 4b
    gamma = 2 pi k / 2b
    where 0 <= i, j, k < 2b are indices
    This grid can be obtained from the function: S3.linspace or S3.meshgrid

    The quadrature weights for this grid are
    w_B(j) = 2/b * sin(pi(2j + 1) / 4b) * sum_{k=0}^{b-1} 1 / (2 k + 1) sin((2j + 1)(2k + 1) pi / 4b)
    This is eq. 23 in [1] and eq. 2.15 in [2].

    [1] SOFT: SO(3) Fourier Transforms
    Peter J. Kostelec and Daniel N. Rockmore

    [2] FFTs on the Rotation Group
    Peter J. Kostelec · Daniel N. Rockmore

    :param b: bandwidth (grid has shape 2b * 2b * 2b)
    :return: w: an array of length 2b containing the quadrature weigths
    """
    if grid_type == 'SOFT':
        k = np.arange(0, b)
        w = np.array([(2. / b) * np.sin(np.pi * (2. * j + 1.) / (4. * b)) *
                      (np.sum((1. / (2 * k + 1))
                              * np.sin((2 * j + 1) * (2 * k + 1)
                                       * np.pi / (4. * b))))
                      for j in range(2 * b)])

        # This is not in the SOFT documentation, but we found that it is necessary to divide by this factor to
        # get correct results.
        w /= 2. * ((2 * b) ** 2)

        # In the SOFT source, they talk about the following weights being used for
        # odd-order transforms. Do not understand this, and the weights used above
        # (defined in the SOFT papers) seems to work.
        # w = np.array([(2. / b) *
        #              (np.sum((1. / (2 * k + 1))
        #                      * np.sin((2 * j + 1) * (2 * k + 1)
        #                               * np.pi / (4. * b))))
        #              for j in range(2 * b)])
        return w
    else:
        raise NotImplementedError
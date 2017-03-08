
import numpy as np


def linspace(b):
    """
    Compute a linspace on the 3-sphere.

    Since S3 is ismorphic to SO(3), we use the grid convention from:
    FFTs on the Rotation Group
    Peter J. Kostelec and Daniel N. Rockmore
    http://www.cs.dartmouth.edu/~geelong/soft/03-11-060.pdf
    :param b:
    :return:
    """
    alpha = 2 * np.pi * np.arange(2 * b) / (2. * b)
    beta = np.pi * (2 * np.arange(2 * b) + 1) / (4. * b)
    gamma = 2 * np.pi * np.arange(2 * b) / (2. * b)
    return alpha, beta, gamma


def meshgrid(b):
    return np.meshgrid(*linspace(b))


def integrate(f, normalize=True):
    """
    Integrate a function f : S^3 -> R over the 3-sphere S^3, using the invariant integration measure
    mu((alpha, beta, gamma)) = dalpha sin(beta) dbeta dgamma
    i.e. this returns
    int_S^3 f(x) dmu(x) = int_0^2pi int_0^pi int_0^2pi f(alpha, beta, gamma) dalpha sin(beta) dbeta dgamma

    :param f: a function of three scalar variables returning a scalar.
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
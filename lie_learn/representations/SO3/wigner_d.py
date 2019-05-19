
import numpy as np

from lie_learn.representations.SO3.pinchon_hoggan.pinchon_hoggan_dense import Jd, rot_mat
from lie_learn.representations.SO3.irrep_bases import change_of_basis_matrix


def wigner_d_matrix(l, beta,
                    field='real', normalization='quantum', order='centered', condon_shortley='cs'):
    """
    Compute the Wigner-d matrix of degree l at beta, in the basis defined by
    (field, normalization, order, condon_shortley)

    The Wigner-d matrix of degree l has shape (2l + 1) x (2l + 1).

    :param l: the degree of the Wigner-d function. l >= 0
    :param beta: the argument. 0 <= beta <= pi
    :param field: 'real' or 'complex'
    :param normalization: 'quantum', 'seismology', 'geodesy' or 'nfft'
    :param order: 'centered' or 'block'
    :param condon_shortley: 'cs' or 'nocs'
    :return: d^l_mn(beta) in the chosen basis
    """
    # This returns the d matrix in the (real, quantum-normalized, centered, cs) convention
    d = rot_mat(alpha=0., beta=beta, gamma=0., l=l, J=Jd[l])

    if (field, normalization, order, condon_shortley) != ('real', 'quantum', 'centered', 'cs'):
        # TODO use change of basis function instead of matrix?
        B = change_of_basis_matrix(
            l,
            frm=('real', 'quantum', 'centered', 'cs'),
            to=(field, normalization, order, condon_shortley))
        BB = change_of_basis_matrix(
            l,
            frm=(field, normalization, order, condon_shortley),
            to=('real', 'quantum', 'centered', 'cs'))
        d = B.dot(d).dot(BB)

        # The Wigner-d matrices are always real, even in the complex basis
        # (I tested this numerically, and have seen it in several texts)
        # assert np.isclose(np.sum(np.abs(d.imag)), 0.0)
        d = d.real

    return d


def wigner_D_matrix(l, alpha, beta, gamma,
                    field='real', normalization='quantum', order='centered', condon_shortley='cs'):
    """
    Evaluate the Wigner-d matrix D^l_mn(alpha, beta, gamma)

    :param l: the degree of the Wigner-d function. l >= 0
    :param alpha: the argument. 0 <= alpha <= 2 pi
    :param beta: the argument. 0 <= beta <= pi
    :param gamma: the argument. 0 <= gamma <= 2 pi
    :param field: 'real' or 'complex'
    :param normalization: 'quantum', 'seismology', 'geodesy' or 'nfft'
    :param order: 'centered' or 'block'
    :param condon_shortley: 'cs' or 'nocs'
    :return: D^l_mn(alpha, beta, gamma) in the chosen basis
    """

    D = rot_mat(alpha=alpha, beta=beta, gamma=gamma, l=l, J=Jd[l])

    if (field, normalization, order, condon_shortley) != ('real', 'quantum', 'centered', 'cs'):
        B = change_of_basis_matrix(
            l,
            frm=('real', 'quantum', 'centered', 'cs'),
            to=(field, normalization, order, condon_shortley))
        BB = change_of_basis_matrix(
            l,
            frm=(field, normalization, order, condon_shortley),
            to=('real', 'quantum', 'centered', 'cs'))
        D = B.dot(D).dot(BB)

        if field == 'real':
            # print('WIGNER D IMAG PART:', np.sum(np.abs(D.imag)))
            assert np.isclose(np.sum(np.abs(D.imag)), 0.0)
            D = D.real

    return D


def wigner_d_function(l, m, n, beta,
                      field='real', normalization='quantum', order='centered', condon_shortley='cs'):
    """
    Evaluate a single Wigner-d function d^l_mn(beta)

    NOTE: for now, we implement this by computing the entire degree-l Wigner-d matrix and then selecting
    the (m,n) element, so this function is not fast.

    :param l: the degree of the Wigner-d function. l >= 0
    :param m: the order of the Wigner-d function. -l <= m <= l
    :param n: the order of the Wigner-d function. -l <= n <= l
    :param beta: the argument. 0 <= beta <= pi
    :param field: 'real' or 'complex'
    :param normalization: 'quantum', 'seismology', 'geodesy' or 'nfft'
    :param order: 'centered' or 'block'
    :param condon_shortley: 'cs' or 'nocs'
    :return: d^l_mn(beta) in the chosen basis
    """
    return wigner_d_matrix(l, beta, field, normalization, order, condon_shortley)[l + m, l + n]


def wigner_D_function(l, m, n, alpha, beta, gamma,
                      field='real', normalization='quantum', order='centered', condon_shortley='cs'):
    """
    Evaluate a single Wigner-d function d^l_mn(beta)

    NOTE: for now, we implement this by computing the entire degree-l Wigner-D matrix and then selecting
    the (m,n) element, so this function is not fast.

    :param l: the degree of the Wigner-d function. l >= 0
    :param m: the order of the Wigner-d function. -l <= m <= l
    :param n: the order of the Wigner-d function. -l <= n <= l
    :param alpha: the argument. 0 <= alpha <= 2 pi
    :param beta: the argument. 0 <= beta <= pi
    :param gamma: the argument. 0 <= gamma <= 2 pi
    :param field: 'real' or 'complex'
    :param normalization: 'quantum', 'seismology', 'geodesy' or 'nfft'
    :param order: 'centered' or 'block'
    :param condon_shortley: 'cs' or 'nocs'
    :return: d^l_mn(beta) in the chosen basis
    """
    return wigner_D_matrix(l, alpha, beta, gamma, field, normalization, order, condon_shortley)[l + m, l + n]


def wigner_D_norm(l, normalized_haar=True):
    """
    Compute the squared norm of the Wigner-D functions.

    The squared norm of a function on the SO(3) is defined as
    |f|^2 = int_SO(3) |f(g)|^2 dg
    where dg is a Haar measure.

    :param l: for some normalization conventions, the norm of a Wigner-D function D^l_mn depends on the degree l
    :param normalized_haar: whether to use the Haar measure da db sinb dc or the normalized Haar measure
     da db sinb dc / 8pi^2
    :return: the squared norm of the spherical harmonic with respect to given measure

    :param l:
    :param normalization:
    :return:
    """
    if normalized_haar:
        return 1. / (2 * l + 1)
    else:
        return (8 * np.pi ** 2) / (2 * l + 1)


def wigner_d_naive(l, m, n, beta):
    """
    Numerically naive implementation of the Wigner-d function.
    This is useful for checking the correctness of other implementations.

    :param l: the degree of the Wigner-d function. l >= 0
    :param m: the order of the Wigner-d function. -l <= m <= l
    :param n: the order of the Wigner-d function. -l <= n <= l
    :param beta: the argument. 0 <= beta <= pi
    :return: d^l_mn(beta) in the TODO: what basis? complex, quantum(?), centered, cs(?)
    """
    from scipy.special import eval_jacobi
    try:
        from scipy.misc import factorial
    except:
        from scipy.special import factorial

    from sympy.functions.special.polynomials import jacobi, jacobi_normalized
    from sympy.abc import j, a, b, x
    from sympy import N
    #jfun = jacobi_normalized(j, a, b, x)
    jfun = jacobi(j, a, b, x)
    # eval_jacobi = lambda q, r, p, o: float(jfun.eval(int(q), int(r), int(p), float(o)))
    # eval_jacobi = lambda q, r, p, o: float(N(jfun, int(q), int(r), int(p), float(o)))
    eval_jacobi = lambda q, r, p, o: float(jfun.subs({j:int(q), a:int(r), b:int(p), x:float(o)}))

    mu = np.abs(m - n)
    nu = np.abs(m + n)
    s = l - (mu + nu) / 2
    xi = 1 if n >= m else (-1) ** (n - m)

    # print(s, mu, nu, np.cos(beta), type(s), type(mu), type(nu), type(np.cos(beta)))
    jac = eval_jacobi(s, mu, nu, np.cos(beta))
    z = np.sqrt((factorial(s) * factorial(s + mu + nu)) / (factorial(s + mu) * factorial(s + nu)))

    # print(l, m, n, beta, np.isfinite(mu), np.isfinite(nu), np.isfinite(s), np.isfinite(xi), np.isfinite(jac), np.isfinite(z))
    assert np.isfinite(mu) and np.isfinite(nu) and np.isfinite(s) and np.isfinite(xi) and np.isfinite(jac) and np.isfinite(z)
    assert np.isfinite(xi * z * np.sin(beta / 2) ** mu * np.cos(beta / 2) ** nu * jac)
    return xi * z * np.sin(beta / 2) ** mu * np.cos(beta / 2) ** nu * jac


def wigner_d_naive_v2(l, m, n, beta):
    """
    Wigner d functions as defined in the SOFT 2.0 documentation.
    When approx_lim is set to a high value, this function appears to give
    identical results to Johann Goetz' wignerd() function.

    However, integration fails: does not satisfy orthogonality relations everywhere...
    """
    from scipy.special import jacobi

    if n >= m:
        xi = 1
    else:
        xi = (-1)**(n - m)

    mu = np.abs(m - n)
    nu = np.abs(n + m)
    s = l - (mu + nu) * 0.5

    sq = np.sqrt((np.math.factorial(s) * np.math.factorial(s + mu + nu))
                 / (np.math.factorial(s + mu) * np.math.factorial(s + nu)))
    sinb = np.sin(beta * 0.5) ** mu
    cosb = np.cos(beta * 0.5) ** nu
    P = jacobi(s, mu, nu)(np.cos(beta))
    return xi * sq * sinb * cosb * P


def wigner_d_naive_v3(l, m, n, approx_lim=1000000):
    """
    Wigner "small d" matrix. (Euler z-y-z convention)
    example:
        l = 2
        m = 1
        n = 0
        beta = linspace(0,pi,100)
        wd210 = wignerd(l,m,n)(beta)

    some conditions have to be met:
         l >= 0
        -l <= m <= l
        -l <= n <= l

    The approx_lim determines at what point
    bessel functions are used. Default is when:
        l > m+10
          and
        l > n+10

    for integer l and n=0, we can use the spherical harmonics. If in
    addition m=0, we can use the ordinary legendre polynomials.
    """
    from scipy.special import jv, legendre, sph_harm, jacobi
    try:
        from scipy.misc import factorial, comb
    except:
        from scipy.special import factorial, comb
    from numpy import floor, sqrt, sin, cos, exp, power
    from math import pi
    from scipy.special import jacobi

    if (l < 0) or (abs(m) > l) or (abs(n) > l):
        raise ValueError("wignerd(l = {0}, m = {1}, n = {2}) value error.".format(l, m, n) \
            + " Valid range for parameters: l>=0, -l<=m,n<=l.")

    if (l > (m + approx_lim)) and (l > (n + approx_lim)):
        #print 'bessel (approximation)'
        return lambda beta: jv(m - n, l * beta)

    if (floor(l) == l) and (n == 0):
        if m == 0:
            #print 'legendre (exact)'
            return lambda beta: legendre(l)(cos(beta))
        elif False:
            #print 'spherical harmonics (exact)'
            a = sqrt(4. * pi / (2. * l + 1.))
            return lambda beta: a * sph_harm(m, l, beta, 0.).conj()

    jmn_terms = {
        l + n : (m - n, m - n),
        l - n : (n - m, 0.),
        l + m : (n - m, 0.),
        l - m : (m - n, m - n),
        }

    k = min(jmn_terms)
    a, lmb = jmn_terms[k]

    b = 2. * l - 2. * k - a

    if (a < 0) or (b < 0):
        raise ValueError("wignerd(l = {0}, m = {1}, n = {2}) value error.".format(l, m, n) \
            + " Encountered negative values in (a,b) = ({0},{1})".format(a,b))

    coeff = power(-1.,lmb) * sqrt(comb(2. * l - k, k + a)) * (1. / sqrt(comb(k + b, b)))

    #print 'jacobi (exact)'
    return lambda beta: coeff \
        * power(sin(0.5*beta),a) \
        * power(cos(0.5*beta),b) \
        * jacobi(k,a,b)(cos(beta))

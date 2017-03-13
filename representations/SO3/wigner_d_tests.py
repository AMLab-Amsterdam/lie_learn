
import numpy as np

from lie_learn.representations.SO3.wigner_d import wigner_D_matrix, wigner_d_matrix, naive_wigner_d, naive_wigner_d_v2, naive_wigner_d_v3
import lie_learn.spaces.S3 as S3

TEST_L_MAX = 2


def check_unitary():
    """
    Check that the Wigner-D matrices are unitary.
    We test every normalization convention and a range of input angles.

    :return:
    """
    for l in range(TEST_L_MAX):
        for field in ('real', 'complex'):
            for normalization in ('quantum', 'seismology'):  # 'geodesy' and 'nfft' are not unitary
                for order in ('centered', 'block'):
                    for condon_shortley in ('cs', 'nocs'):
                        for a in np.linspace(0, 2 * np.pi, 10):
                            for b in np.linspace(0, np.pi, 10):
                                for c in np.linspace(0, 2 * np.pi, 10):
                                    m = wigner_D_matrix(l, a, b, c, field, normalization, order, condon_shortley)
                                    diff = np.abs(m.conj().T.dot(m) - np.eye(m.shape[0])).sum()
                                    diff += np.abs(m.dot(m.conj().T) - np.eye(m.shape[0])).sum()
                                    print(l, field, normalization, order, condon_shortley, a, b, c, diff)
                                    assert np.isclose(diff, 0.)


def check_normalization_wigner_D():
    """
    According to [1], the Wigner D functions satisfy:

    int_0^2pi da int_0^pi db sin(b) int_0^2pi D^l_mn(a,b,c) D^l_mn(a,b,c)* = 8 pi^2 / (2l+1)

    The factor 8 pi^2 is removed if we integrate with respect to the normalized Haar measure.
    The normalization constant 1 / (2l + 1) can vary depending on the convention.

    Here we test this equality by numerical integration.
    :return:
    """
    # The squared L2 norm for each of the normalizations
    # By L2 norm of f we mean int_SO(3) |f(g)|^2 dg, where dg is the normalized Haar measure
    L2_norm = {
        'quantum': lambda l: 1. / (2 * l + 1),
        'seismology': lambda l: 1. / (2 * l + 1),
        'geodesy': lambda l: (16 * np.pi ** 2) / (2 * l + 1),
        'nfft': lambda l: (16 * np.pi ** 2) / ((2 * l + 1) ** 3)
    }

    for l in range(TEST_L_MAX):
        for m in range(-l, l + 1):
            for n in range(-l, l + 1):
                for field in ('real', 'complex'):
                    for normalization in ('quantum', 'seismology', 'geodesy', 'nfft'):
                        for order in ('centered', 'block'):
                            for condon_shortley in ('cs', 'nocs'):

                                f = lambda a, b, c: np.abs(wigner_D_matrix(
                                    l=l, alpha=a, beta=b, gamma=c,
                                    field=field, normalization=normalization,
                                    order=order, condon_shortley=condon_shortley)[m, n]) ** 2

                                val = S3.integrate(f, normalize=True)
                                print(l, m, n, field, normalization, order, condon_shortley, val, L2_norm[normalization](l))
                                assert np.isclose(val, L2_norm[normalization](l))


def check_orthogonality_wigner_D():
    """
    According to [1], the Wigner D functions satisfy:

    int_0^2pi da int_0^pi db sin(b) int_0^2pi D^l_mn(a,b,c) D^l'_m'n'(a,b,c)*
     =
      8 pi^2 / (2l+1) delta(ll') delta(mm') delta(nn')

    The factor 8 pi^2 is removed if we integrate with respect to the normalized Haar measure.
    The normalization constant 1 / (2l + 1) can vary depending on the convention.

    Here we test this equality by numerical integration.

    Warning: thus function takes a long time to run.
    :return:
    """
    # Note: this function is called "check" not "test" because this function is expensive to evaluate and we don't
    # want to automatically call this when running nosetests.

    # The squared L2 norm for each of the normalizations
    # By L2 norm of f we mean int_SO(3) |f(g)|^2 dg, where dg is the normalized Haar measure
    L2_norm = {
        'quantum': lambda l: 1. / (2 * l + 1),
        'seismology': lambda l: 1. / (2 * l + 1),
        'geodesy': lambda l: (16 * np.pi ** 2) / (2 * l + 1),
        'nfft': lambda l: (16 * np.pi ** 2) / ((2 * l + 1) ** 3)
    }

    for field in ('real', 'complex'):
        for normalization in ('quantum', 'seismology', 'geodesy', 'nfft'):
            for order in ('centered', 'block'):
                for condon_shortley in ('cs', 'nocs'):
                    for l in range(TEST_L_MAX):
                        for m in range(-l, l + 1):
                            for n in range(-l, l + 1):
                                for l2 in range(TEST_L_MAX):
                                    for m2 in range(-l2, l2 + 1):
                                        for n2 in range(-l2, l2 + 1):
                                            f = lambda a, b, c: wigner_D_matrix(
                                                    l=l, alpha=a, beta=b, gamma=c,
                                                    field=field, normalization=normalization,
                                                    order=order, condon_shortley=condon_shortley)[m, n] * \
                                                wigner_D_matrix(
                                                    l=l2, alpha=a, beta=b, gamma=c,
                                                    field=field, normalization=normalization,
                                                    order=order, condon_shortley=condon_shortley)[m2, n2].conj()

                                            val = S3.integrate(f, normalize=True)
                                            print(field, normalization, order, condon_shortley, l, m, n, l2, m2, n2,
                                                  np.round(val, 2),
                                                  np.round(L2_norm[normalization](l) * (l == l2) * (m == m2) * (n == n2), 2))
                                            assert np.isclose(val, L2_norm[normalization](l) * (l == l2) * (m == m2) * (n == n2))










def check_sympy_jacobi_polynomial():

    from sympy.functions.special.polynomials import jacobi, jacobi_normalized
    from sympy.abc import j, a, b, x
    # jfun = jacobi_normalized(j, a, b, x)
    jfun = jacobi_normalized(j, a, b, x)
    # eval_jacobi = lambda q, r, p, o: float(jfun.eval(int(q), int(r), int(p), float(o)))
    # eval_jacobi = lambda q, r, p, o: float(N(jfun, int(q), int(r), int(p), float(o)))
    eval_jacobi = lambda jj, aa, bb, xx: float(jfun.subs({j: int(jj), a: int(aa), b: int(bb), x: float(xx)}))


    f = lambda xx: eval_jacobi(2, 1, 1, xx) * eval_jacobi(2, 1, 1, xx) * (1-xx) ** 1 * (1+xx) ** 1
    from scipy.integrate import quad
    val = quad(f, a=-1, b=1)[0]
    print(val)


def myquad(f, a, b):
    n = 10000
    v = 0.
    for x in np.linspace(a, b, num=n, endpoint=False):
        v += f(x)
    return v * (b - a) / n


def check_normalization_wigner_d():
    """
    According to [1], the following is true (eq. 12)
    int_0^pi d^l_mn(beta) d^l'_m'n'(beta) sin(beta) dbeta = 1 / (2 l + 1) delta(l,l')

    [1] SOFT: SO(3) Fourier Transforms
    Peter J. Kostelec and Daniel N. Rockmore
    :return:
    """
    # Note: this function is called "check" not "test" because this function is expensive to evaluate and we don't
    # want to automatically call this when running nosetests.

    # The squared L2 norm for each of the normalizations
    # By L2 norm of f we mean int_SO(3) |f(g)|^2 dg, where dg is the normalized Haar measure
    L2_norm = {
        'quantum': lambda l: 1. / (2 * l + 1),
        'seismology': lambda l: 1. / (2 * l + 1),
        'geodesy': lambda l: (16 * np.pi ** 2) / (2 * l + 1),
        'nfft': lambda l: (16 * np.pi ** 2) / ((2 * l + 1) ** 3)
    }
    L2_norm = {
        'quantum': lambda l: 2. / (2 * l + 1),
        'seismology': lambda l: 2. / (2 * l + 1),
        'geodesy': lambda l: 2. / (2 * l + 1),
        'nfft': lambda l: 2. / (2 * l + 1)
    }


    for l in range(TEST_L_MAX):
        for m in range(-l, l + 1):
            for n in range(-l, l + 1):
                for field in ('real', 'complex'):
                    for normalization in ('quantum', 'seismology', 'geodesy', 'nfft'):
                        for order in ('centered', 'block'):
                            for condon_shortley in ('cs', 'nocs'):

                                f = lambda b: wigner_d_matrix(
                                    l=l, beta=b,
                                    field=field, normalization=normalization,
                                    order=order, condon_shortley=condon_shortley)[m, n] ** 2 * np.sin(b)

                                # from scipy.integrate import quad
                                # res = quad(f, a=0, b=np.pi, full_output=1)
                                # val = res[0]
                                # if not np.isclose(val, L2_norm[normalization](l)):
                                #     print(res)
                                val = myquad(f, 0, np.pi)
                                print(l, m, n, field, normalization, order, condon_shortley, val, L2_norm[normalization](l))
                                # assert np.isclose(val, L2_norm[normalization](l))


def check_normalization_naive_wigner_d():
    """
    According to [1], the following is true (eq. 12)
    int_0^pi d^l_mn(beta) d^l'_m'n'(beta) sin(beta) dbeta = 1 / (2 l + 1) delta(l,l')

    [1] SOFT: SO(3) Fourier Transforms
    Peter J. Kostelec and Daniel N. Rockmore
    :return:
    """
    # Note: this function is called "check" not "test" because this function is expensive to evaluate and we don't
    # want to automatically call this when running nosetests.

    # The squared L2 norm for each of the normalizations
    # By L2 norm of f we mean int_SO(3) |f(g)|^2 dg, where dg is the normalized Haar measure
    L2_norm = {
        'quantum': lambda l: 1. / (2 * l + 1),
        'seismology': lambda l: 1. / (2 * l + 1),
        'geodesy': lambda l: (16 * np.pi ** 2) / (2 * l + 1),
        'nfft': lambda l: (16 * np.pi ** 2) / ((2 * l + 1) ** 3)
    }

    for l in range(TEST_L_MAX):
        for m in range(-l, l + 1):
            for n in range(-l, l + 1):
                for l2 in range(TEST_L_MAX):
                    for m2 in range(-l2, l2 + 1):
                        for n2 in range(-l2, l2 + 1):
                                #f = lambda b: naive_wigner_d_v2(l=l, m=m, n=n, beta=b)\
                                #              * naive_wigner_d_v2(l=l2, m=m2, n=n2, beta=b) * np.sin(b)
                                f = lambda b: naive_wigner_d_v3(l=l, m=m, n=n)(b) \
                                              * naive_wigner_d_v3(l=l2, m=m2, n=n2)(b) * np.sin(b)
                                # from scipy.integrate import quad
                                # val = quad(f, a=0, b=np.pi)[0]
                                val = myquad(f, 0, np.pi)

                                print(l, m, n, l2, m2, n2, val, 2 / (2 * l + 1) * (l == l2))
                                # assert np.isclose(val, L2_norm[normalization](l))


def make_wigfun(field, normalization, order, cs):
    return lambda l, m, n, b: wigner_d_matrix(l, b, field, normalization, order, cs)[m, n]

def check_orthogonality_wigner_d(wigfun):
    for l in range(TEST_L_MAX):
        for m in range(-l, l + 1):
            for n in range(-l, l + 1):
                for l2 in range(TEST_L_MAX):
                    for m2 in range(-l2, l2 + 1):
                        for n2 in range(-l2, l2 + 1):
                            def f(b):
                                w1 = wigfun(l, m, n, b)
                                w2 = wigfun(l2, m2, n2, b)
                                return w1 * w2 * np.sin(b)

                            from scipy.integrate import quad
                            val = quad(f, a=0, b=np.pi)[0]
                            print(l, m, n, l2, m2, n2, np.round(val, 3), int(l == l2))
                            # assert np.isclose(val, L2_norm[normalization](l))


def check_orthogonality_wigner_d_all():
    """
    According to [1], the following is true (eq. 12)
    int_0^pi d^l_mn(beta) d^l'_m'n'(beta) sin(beta) dbeta = 1 / (2 l + 1) delta(l,l')

    [1] SOFT: SO(3) Fourier Transforms
    Peter J. Kostelec and Daniel N. Rockmore
    :return:
    """
    # Note: this function is called "check" not "test" because this function is expensive to evaluate and we don't
    # want to automatically call this when running nosetests.

    # The squared L2 norm for each of the normalizations
    # By L2 norm of f we mean int_SO(3) |f(g)|^2 dg, where dg is the normalized Haar measure
    L2_norm = {
        'quantum': lambda l: 1. / (2 * l + 1),
        'seismology': lambda l: 1. / (2 * l + 1),
        'geodesy': lambda l: (16 * np.pi ** 2) / (2 * l + 1),
        'nfft': lambda l: (16 * np.pi ** 2) / ((2 * l + 1) ** 3)
    }

    for field in ('real', 'complex'):
        for normalization in ('quantum', 'seismology', 'geodesy', 'nfft'):
            for order in ('centered', 'block'):
                for condon_shortley in ('cs', 'nocs'):
                    for l in range(TEST_L_MAX):
                        for m in range(-l, l + 1):
                            for n in range(-l, l + 1):
                                for l2 in range(TEST_L_MAX):
                                    for m2 in range(-l2, l2 + 1):
                                        for n2 in range(-l2, l2 + 1):

                                            def f(b):
                                                w1 = wigner_d_matrix(
                                                    l=l, beta=b,
                                                    field=field, normalization=normalization,
                                                    order=order, condon_shortley=condon_shortley)[m, n]
                                                w2 = wigner_d_matrix(
                                                    l=l2, beta=b,
                                                    field=field, normalization=normalization,
                                                    order=order, condon_shortley=condon_shortley)[m2, n2]
                                                return w1 * w2 * np.sin(b)

                                            from scipy.integrate import quad
                                            val = quad(f, a=0, b=np.pi)[0]
                                            print(field, normalization, order, condon_shortley, l, m, n, l2, m2, n2,
                                                  np.round(val, 2), L2_norm[normalization](l) * (l == l2))
                                            # assert np.isclose(val, L2_norm[normalization](l))


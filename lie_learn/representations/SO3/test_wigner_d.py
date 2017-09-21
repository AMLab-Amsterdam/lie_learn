
import numpy as np

from lie_learn.representations.SO3.wigner_d import wigner_D_matrix, wigner_d_matrix,\
    wigner_d_naive, wigner_d_naive_v2, wigner_d_naive_v3, wigner_d_function, wigner_D_function
import lie_learn.spaces.S3 as S3

TEST_L_MAX = 3

def check_unitarity_wigner_D():
    """
    Check that the Wigner-D matrices are unitary.
    We test every normalization convention and a range of input angles.

    Note: only the quantum- or seismology normalized Wigner-D matrices are unitary,
    so we do not check the geodesy and nfft normalized matrices.
    """
    for l in range(TEST_L_MAX):
        for field in ('real', 'complex'):
            for normalization in ('quantum', 'seismology', 'geodesy', 'nfft'):
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

    int_0^2pi da int_0^pi db sin(b) int_0^2pi |D^l_mn(a,b,c)|^2 = 8 pi^2 / (2l+1)

    The factor 8 pi^2 is removed if we integrate with respect to the normalized Haar measure.

    Here we test this equality by numerical integration.

    NOTE: this test is subsumed in check_orthogonality_wigner_D, but that function is very slow
    """
    w = S3.quadrature_weights(b=TEST_L_MAX + 1, grid_type='SOFT')
    for l in range(TEST_L_MAX):
        for m in range(-l, l + 1):
            for n in range(-l, l + 1):
                for field in ('real', 'complex'):
                    for normalization in ('quantum', 'seismology', 'geodesy', 'nfft'):
                        for order in ('centered', 'block'):
                            for condon_shortley in ('cs', 'nocs'):

                                f = lambda a, b, c: np.abs(wigner_D_function(
                                    l=l, m=m, n=n, alpha=a, beta=b, gamma=c,
                                    field=field, normalization=normalization,
                                    order=order, condon_shortley=condon_shortley)) ** 2
                                sqnorm_numerical = S3.integrate(f, normalize=True)

                                D = make_D_sample_grid(b=TEST_L_MAX + 1, l=l, m=m, n=n,
                                                       field=field, normalization=normalization,
                                                       order=order, condon_shortley=condon_shortley)
                                sqnorm_numerical2 = S3.integrate_quad(D * D.conj(), grid_type='SOFT',
                                                                      normalize=True, w=w)

                                sqnorm_analytical = 1. / (2 * l + 1)
                                print(l, m, n, field, normalization, order, condon_shortley, sqnorm_numerical, sqnorm_numerical2, sqnorm_analytical)
                                assert np.isclose(sqnorm_numerical, sqnorm_analytical)
                                assert np.isclose(sqnorm_numerical2, sqnorm_analytical)


def check_orthogonality_wigner_D():
    """
    According to [1], the Wigner D functions satisfy:

    int_0^2pi da int_0^pi db sin(b) int_0^2pi D^l_mn(a,b,c) D^l'_m'n'(a,b,c)*
     =
      8 pi^2 / (2l+1) delta(ll') delta(mm') delta(nn')

    The factor 8 pi^2 is removed if we integrate with respect to the normalized Haar measure.
    Here we test this equality by numerical integration.
    """
    w = S3.quadrature_weights(b=TEST_L_MAX + 1, grid_type='SOFT')
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

                                            f = lambda a, b, c:\
                                                wigner_D_function(
                                                    l=l, m=m, n=n, alpha=a, beta=b, gamma=c,
                                                    field=field, normalization=normalization,
                                                    order=order, condon_shortley=condon_shortley) * \
                                                wigner_D_function(
                                                    l=l2, m=m2, n=n2, alpha=a, beta=b, gamma=c,
                                                    field=field, normalization=normalization,
                                                    order=order, condon_shortley=condon_shortley).conj()

                                            D1 = make_D_sample_grid(b=TEST_L_MAX + 1, l=l, m=m, n=n,
                                                                    field=field, normalization=normalization,
                                                                    order=order, condon_shortley=condon_shortley)
                                            D2 = make_D_sample_grid(b=TEST_L_MAX + 1, l=l2, m=m2, n=n2,
                                                                    field=field, normalization=normalization,
                                                                    order=order, condon_shortley=condon_shortley)

                                            numerical_norm2 = S3.integrate_quad(D1 * D2.conj(), grid_type='SOFT',
                                                                                normalize=True, w=w)
                                            numerical_norm = S3.integrate(f, normalize=True)
                                            analytical_norm = ((l == l2) * (m == m2) * (n == n2)) / (2 * l + 1)
                                            print(field, normalization, order, condon_shortley, l, m, n, l2, m2, n2,
                                                  np.round(numerical_norm, 2),
                                                  np.round(numerical_norm2, 2),
                                                  np.round(analytical_norm, 2))
                                            assert np.isclose(numerical_norm, analytical_norm)
                                            assert np.isclose(numerical_norm2, analytical_norm)


def check_normalization_complex_wigner_d():
    """
    According to [1], the following is true (eq. 12)
    int_0^pi |d^l_mn(beta)|^2 sin(beta) dbeta = 1 / (2 l + 1)

    NOTE: this function only tests the Wigner-d functions in the *complex basis*.
    In this basis, the Wigner-d functions all have the same, simple norm: 2. / (2l + 1)
    In the real basis, some functions are identically 0 and for the rest the norm is hard to understand.
    We treat these in a separate function below.

    [1] SOFT: SO(3) Fourier Transforms
    Peter J. Kostelec and Daniel N. Rockmore
    """
    # The squared L2 norm
    # By squared L2 norm of f we mean |f|^2 = int_SO(3) |f(g)|^2 dg, where dg is the normalized Haar measure
    L2_norm = lambda l: 2. / (2 * l + 1)  # Note the factor 2..

    for l in range(TEST_L_MAX):
        for m in range(-l, l + 1):
            for n in range(-l, l + 1):
                for field in ('complex',):  # Only test complex d functions here
                    for normalization in ('quantum', 'seismology', 'geodesy', 'nfft'):
                        for condon_shortley in ('cs', 'nocs'):
                            for order in ('centered', 'block'):

                                f = lambda b: wigner_d_matrix(
                                    l=l, beta=b,
                                    field=field, normalization=normalization,
                                    order=order, condon_shortley=condon_shortley)[l + m, l + n] ** 2 * np.sin(b)

                                # from scipy.integrate import quad
                                # res = quad(f, a=0, b=np.pi, full_output=1)
                                # val = res[0]
                                # if not np.isclose(val, L2_norm[normalization](l)):
                                #     print(res)
                                val = myquad(f, 0, np.pi)

                                print(l, m, n, field, normalization, order, condon_shortley,
                                      np.round(val, 2),
                                      np.round(L2_norm(l), 2))
                                assert np.isclose(val, L2_norm(l))


def check_orthogonality_complex_wigner_d():
    """
    According to [1], the following is true (eq. 12)
    int_0^pi d^l_mn(beta) d^l'_mn(beta) sin(beta) dbeta = 1 / (2 l + 1) delta(l,l')

    NOTE: this function only tests the Wigner-d functions in the *complex basis*.
    In this basis, the Wigner-d functions all have the same, simple norm: 2. / (2l + 1)
    In the real basis, some functions are identically 0 and for the rest the norm is hard to understand.
    We treat these in a separate function below.

    NOTE: we only test in centered, not the block basis. For some reason this equality fails in the block basis.
    I have not investigated the reason for this yet.

    [1] SOFT: SO(3) Fourier Transforms
    Peter J. Kostelec and Daniel N. Rockmore
    :return:
    """
    # The squared L2 norm for each of the normalizations
    # By squared L2 norm of f we mean |f|^2 = int_SO(3) |f(g)|^2 dg, where dg is the normalized Haar measure
    L2_norm = lambda l: 2. / (2 * l + 1)

    for field in ('complex',):
        for normalization in ('quantum', 'seismology', 'geodesy', 'nfft'):
            for condon_shortley in ('cs', 'nocs'):
                for order in ('centered',):  # 'block'):
                    for m in range(-TEST_L_MAX, TEST_L_MAX + 1):
                        for n in range(-TEST_L_MAX, TEST_L_MAX + 1):
                            for l in range(np.maximum(np.abs(m), np.abs(n)), TEST_L_MAX):
                                for l2 in range(np.maximum(np.abs(m), np.abs(n)), TEST_L_MAX):

                                    f = lambda b:\
                                        wigner_d_function(
                                            l=l, m=m, n=n, beta=b,
                                            field=field, normalization=normalization,
                                            order=order, condon_shortley=condon_shortley) * \
                                        wigner_d_function(
                                            l= l2, m=m, n=n, beta=b,
                                            field=field, normalization=normalization,
                                            order=order, condon_shortley=condon_shortley) * \
                                        np.sin(b)

                                    # from scipy.integrate import quad
                                    # res = quad(f, a=0, b=np.pi, full_output=1)
                                    # val = res[0]
                                    # if not np.isclose(val, L2_norm[normalization](l)):
                                    #     print(res)
                                    numerical_inner_product = myquad(f, 0, np.pi)
                                    analytical_inner_product = L2_norm(l) * (l == l2)

                                    print(l, l2, m, n, field, normalization, order, condon_shortley,
                                          np.round(numerical_inner_product, 2),
                                          np.round(analytical_inner_product, 2))
                                    assert np.isclose(numerical_inner_product, analytical_inner_product,
                                                      rtol=1e-4, atol=1e-5)


def check_orthogonality_naive_wigner_d():
    """
    According to [1], the following is true (eq. 12)
    int_0^pi d^l_mn(beta) d^l'_mn(beta) sin(beta) dbeta = 1 / (2 l + 1) delta(l, l')

    Here we check this equality numerically for the *naive* implementations of the Wigner-d functions

    [1] SOFT: SO(3) Fourier Transforms
    Peter J. Kostelec and Daniel N. Rockmore
    :return:
    """

    # The squared L2 norm for each of the normalizations
    # By squared L2 norm of f we mean |f|^2 = int_SO(3) |f(g)|^2 dg, where dg is the normalized Haar measure
    L2_norm = lambda l: 2. / (2 * l + 1)

    for m in range(-TEST_L_MAX, TEST_L_MAX + 1):
        for n in range(-TEST_L_MAX, TEST_L_MAX + 1):
            for l in range(np.maximum(np.abs(m), np.abs(n)), TEST_L_MAX):
                for l2 in range(np.maximum(np.abs(m), np.abs(n)), TEST_L_MAX):

                    f1 = lambda b: \
                        wigner_d_naive(l=l, m=m, n=n, beta=b) * \
                        wigner_d_naive(l=l2, m=m, n=n, beta=b) * \
                        np.sin(b)

                    f2 = lambda b: \
                        wigner_d_naive_v2(l=l, m=m, n=n, beta=b) * \
                        wigner_d_naive_v2(l=l2, m=m, n=n, beta=b) * \
                        np.sin(b)

                    f3 = lambda b: \
                        wigner_d_naive_v3(l=l, m=m, n=n)(b) * \
                        wigner_d_naive_v3(l=l2, m=m, n=n)(b) * \
                        np.sin(b)

                    for f in (f1, f2, f3):

                        # from scipy.integrate import quad
                        # res = quad(f, a=0, b=np.pi, full_output=1)
                        # val = res[0]
                        # if not np.isclose(val, L2_norm[normalization](l)):
                        #     print(res)
                        numerical_inner_product = myquad(f, 0, np.pi)
                        analytical_inner_product = L2_norm(l) * (l == l2)

                        print(l, l2, m, n,
                              np.round(numerical_inner_product, 2),
                              np.round(analytical_inner_product, 2))
                        assert np.isclose(numerical_inner_product, analytical_inner_product,
                                          rtol=1e-4, atol=1e-5)


def myquad(f, a, b):

    n = 1000
    v = 0.
    for x in np.linspace(a, b, num=n, endpoint=False):
        v += f(x)
    return v * (b - a) / n


# TODO: this test is failing - I'm not sure what the norms for real Wigner-d functions should be (see comments below)
def check_normalization_wigner_d_real(L_max=TEST_L_MAX):
    """
    According to [1], the following is true (eq. 12)
    int_0^pi d^l_mn(beta) d^l'_mn(beta) sin(beta) dbeta = 1 / (2 l + 1) delta(l, l')

    [1] SOFT: SO(3) Fourier Transforms
    Peter J. Kostelec and Daniel N. Rockmore
    :return:
    """
    # Note: this function is called "check" not "test" because this function is expensive to evaluate and we don't
    # want to automatically call this when running nosetests.

    # The squared L2 norm for each of the normalizations
    # By L2 norm of f we mean int_SO(3) |f(g)|^2 dg, where dg is the normalized Haar measure
    L2_norm = {
        'quantum': lambda l: 2. / (2 * l + 1),
        'seismology': lambda l: 2. / (2 * l + 1),
        'geodesy': lambda l: 2. / (2 * l + 1),
        'nfft': lambda l: 2. / (2 * l + 1)
    }
    correct = [np.zeros((2 * l + 1, 2 * l + 1)) for l in range(L_max)]
    ratio = [np.zeros((2 * l + 1, 2 * l + 1)) for l in range(L_max)]
    vals = [np.zeros((2 * l + 1, 2 * l + 1)) for l in range(L_max)]

    # Note: this seems to be correct for complex wigners in all normalizations, orders, cs, l, m, n,
    # For the real ones, we can understand which wigners are identically zero,
    # the norms for the non-zeros appear to be pretty complicated. Plotting the norms for l=9, we see a band pattern
    # similar to the appearance of the wigner-d matrix itself.
    # This matrix is symmetric, so the norm for dmn equals the norm for dnm

    # See note above in check_normalization_wigner_d_complex
    # The norm of the real wigner-d functions seems to be hard to understand. The norm now depends on m,n as well as l
    # We can understand which real wigners are identically zero (see real_zeros below, or plot a d-matrix).
    # Plotting the norms for l=9, we see a moire-like pattern for the non-zero wigners,
    # similar in appearance to the wigner-d matrix itself.
    # This matrix is symmetric, so the norm for dmn equals the norm for dnm
    for order in ('centered',):  # 'block'):
        for l in range(L_max):
            for m in range(-l, l + 1):
                for n in range(-l, l + 1):
                    for field in ('real',):  # only test real here
                        for normalization in ('quantum',):  # 'seismology', 'geodesy', 'nfft'): all normalization seem to give the same behaviour
                            for condon_shortley in ('cs',):  # 'nocs'): doesn't seem to matter

                                f = lambda b: wigner_d_matrix(
                                    l=l, beta=b,
                                    field=field, normalization=normalization,
                                    order=order, condon_shortley=condon_shortley)[l + m, l + n] ** 2 * np.sin(b)

                                # from scipy.integrate import quad
                                # res = quad(f, a=0, b=np.pi, full_output=1)
                                # val = res[0]
                                # if not np.isclose(val, L2_norm[normalization](l)):
                                #     print(res)
                                val = myquad(f, 0, np.pi)

                                real_zeros = ((m < 0 and n >= 0) or (m >= 0 and n < 0)) and field == 'real'
                                print(l, m, n, # field, normalization, order, condon_shortley,
                                      np.round(val, 2),
                                      np.round(L2_norm[normalization](l) * (not real_zeros), 2))
                                # assert np.isclose(val, L2_norm[normalization](l))
                                # if not np.isclose(val, L2_norm[normalization](l)):
                                #     print("!!!!!")
                                correct[l][l + m, l + n] = np.isclose(val, L2_norm[normalization](l) * (not real_zeros))
                                ratio[l][l + m, l + n] = val / (L2_norm[normalization](l) * (not real_zeros)) if (not real_zeros) else 1
                                vals[l][l + m, l + n] = val

    return correct, ratio, vals


def make_D_sample_grid(b=4, l=0, m=0, n=0,
                       field='complex', normalization='seismology', order='centered', condon_shortley='cs'):

    from lie_learn.representations.SO3.wigner_d import wigner_D_function
    D = lambda a, b, c: wigner_D_function(l, m, n, alpha, beta, gamma,
                                          field=field, normalization=normalization,
                                          order=order, condon_shortley=condon_shortley)

    f = np.zeros((2 * b, 2 * b, 2 * b), dtype=complex if field == 'complex' else float)

    for j1 in range(f.shape[0]):
        alpha = 2 * np.pi * j1 / (2. * b)
        for k in range(f.shape[1]):
            beta = np.pi * (2 * k + 1) / (4. * b)
            for j2 in range(f.shape[2]):
                gamma = 2 * np.pi * j2 / (2. * b)
                f[j1, k, j2] = D(alpha, beta, gamma)
    return f

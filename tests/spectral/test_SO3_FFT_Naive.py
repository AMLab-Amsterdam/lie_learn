
import numpy as np
from lie_learn.spectral.SO3FFT_Naive import SO3_FFT_NaiveReal, SO3_FFT_SemiNaive_Complex, SO3_FT_Naive
from lie_learn.representations.SO3.pinchon_hoggan.pinchon_hoggan_dense import Jd, rot_mat
from lie_learn.representations.SO3.irrep_bases import change_of_basis_matrix


# TODO: test if the Fourier transform of a right SO(2)-invariant function is zero except for a column at n=0, and
# test if it is equal to the spherical harmonics transform of the corresponding function on the sphere


def test_SO3_FT_Naive():
    """
    Check that the naive complex SO(3) FFT:
    - Produces the right Wigner-D function when given a 1-hot input to the synthesis transform
    - Produces a 1-hot vector when given a single Wigner-D function to the analysis transform
    """
    L_max = 3

    f_hat = [np.zeros((2 * ll + 1, 2 * ll + 1)) for ll in range(L_max + 1)]

    # TODO: the SO3_FFT_SemiNaive_Complex no longer uses the D convention parameters because of new caching feature

    field = 'complex'
    order = 'centered'
    for normalization in ('quantum', 'seismology'):  # Note: the geodesy and nfft wigners are normalized differently
        for condon_shortley in ('cs', 'nocs'):

            fft = SO3_FT_Naive(L_max=L_max,
                               field=field, normalization=normalization,
                               order=order, condon_shortley=condon_shortley)

            for l in range(L_max + 1):
                for m in range(-l, l + 1):
                    for n in range(-l, l + 1):
                        f_hat[l][l + m, l + n] = 1. / (2 * l + 1)
                        f_hat_flat = np.hstack([fhl.flatten() for fhl in f_hat])
                        D = fft.synthesize_by_matmul(f_hat_flat)

                        D2 = make_D_sample_grid(b=L_max + 1, l=l, m=m, n=n,
                                                field=field, normalization=normalization,
                                                order=order, condon_shortley=condon_shortley)

                        diff = np.sum(np.abs(D - D2.flatten()))
                        print(l, m, n, 'Synthesize error:', diff)
                        assert np.isclose(diff, 0.0)

                        f_hat_2 = fft.analyze_by_matmul(D2)

                        # f_hat_flat = np.hstack([ff.flatten() for ff in f_hat])
                        f_hat_2_flat = np.hstack([ff.flatten() for ff in f_hat_2])

                        # f_hat_2_flat *= (2 * l + 1)  # / (4 * np.pi)  # apply magic constant TODO fix this
                        print(f_hat_2_flat)
                        print(f_hat_flat)
                        print(np.max(np.abs(f_hat_flat)), np.max(np.abs(f_hat_2_flat)))

                        diff = np.sum(np.abs(f_hat_flat - f_hat_2_flat))
                        print(l, m, n, 'Analyze error:', diff)
                        assert np.isclose(diff, 0.0)

                        f_hat[l][l + m, l + n] = 0.

def test_SO3_FFT_SemiNaiveComplex():
    """
    Check that the naive complex SO(3) FFT:
    - Produces the right Wigner-D function when given a 1-hot input to the synthesis transform
    - Produces a 1-hot vector when given a single Wigner-D function to the analysis transform
    """
    L_max = 3

    f_hat = [np.zeros((2 * ll + 1, 2 * ll + 1)) for ll in range(L_max + 1)]

    # TODO: the SO3_FFT_SemiNaive_Complex no longer uses the D convention parameters because of new caching feature

    field = 'complex'
    order = 'centered'
    for normalization in ('quantum', 'seismology'):  # Note: the geodesy and nfft wigners are normalized differently
        for condon_shortley in ('cs', 'nocs'):

            fft = SO3_FFT_SemiNaive_Complex(L_max=L_max, L2_normalized=False,
                                            field=field, normalization=normalization,
                                            order=order, condon_shortley=condon_shortley)

            #fft = SO3_FFT_Naive(L_max=L_max,
            #                    field=field, normalization=normalization,
            #                    order=order, condon_shortley=condon_shortley)

            for l in range(L_max + 1):
                for m in range(-l, l + 1):
                    for n in range(-l, l + 1):
                        f_hat[l][l + m, l + n] = 1.
                        D = fft.synthesize(f_hat)

                        D2 = make_D_sample_grid(b=L_max + 1, l=l, m=m, n=n,
                                                field=field, normalization=normalization,
                                                order=order, condon_shortley=condon_shortley)

                        diff = np.sum(np.abs(D - D2))
                        print(l, m, n, diff)
                        assert np.isclose(diff, 0.0)

                        f_hat_2 = fft.analyze(D2)

                        f_hat_flat = np.hstack([ff.flatten() for ff in f_hat])
                        f_hat_2_flat = np.hstack([ff.flatten() for ff in f_hat_2])

                        f_hat_2_flat *= (2 * l + 1) / (4 * np.pi)  # apply magic constant TODO fix this

                        diff = np.sum(np.abs(f_hat_flat - f_hat_2_flat))
                        print(l, m, n, diff)
                        assert np.isclose(diff, 0.0)

                        f_hat[l][l + m, l + n] = 0.


# TODO: test linearity of FFT

#TODO
def check_SO3_FFT_NaiveComplex_invertible():
    L_max = 3

    f_hat = [np.zeros((2 * ll + 1, 2 * ll + 1)) for ll in range(L_max + 1)]
    fft = SO3_FFT_SemiNaive_Complex(L_max=L_max, L2_normalized=False)
    for l in range(L_max + 1):
        for m in range(-l, l + 1):
            for n in range(-l, l + 1):
                f_hat[l][l + m, l + n] = 1.
                f = fft.synthesize(f_hat)
                f_hat_2 = fft.analyze(f)

                diff = np.sum([np.abs(f_hat[ll] - f_hat_2[ll]) for ll in range(L_max + 1)])

                f_hat[l][l + m, l + n] = 0.
                print(l, m, n, diff)  # , D2 / D
                assert np.isclose(diff, 0.0)


def test_SO3_FFT_NaiveReal():
    """
    Testing if the real Naive SO(3) FFT synthesis works correctly for 1-hot input vectors
    """
    L_max = 3

    f_hat = [np.zeros((2 * ll + 1, 2 * ll + 1)) for ll in range(L_max + 1)]
    fft = SO3_FFT_NaiveReal(L_max=L_max, L2_normalized=False)
    for l in range(L_max + 1):
        for m in range(-l, l + 1):
            for n in range(-l, l + 1):
                f_hat[l][l + m, l + n] = 1.
                D = fft.synthesize(f_hat)
                f_hat[l][l + m, l + n] = 0.
                D2 = make_D_sample_grid(b=L_max + 1, l=l, m=m, n=n,
                                        field='real', normalization='quantum', order='centered', condon_shortley='cs')

                print(l, m, n, np.sum(np.abs(D - D2)))
                assert np.isclose(np.sum(np.abs(D - D2)), 0.0)


def make_D_sample_grid(b=4, l=0, m=0, n=0,
                       field='complex', normalization='seismology', order='centered', condon_shortley='cs'):

    from lie_learn.representations.SO3.wigner_d import wigner_D_function
    D = lambda a, b, c: wigner_D_function(l, m, n, alpha, beta, gamma,
                                          field=field, normalization=normalization,
                                          order=order, condon_shortley=condon_shortley)

    f = np.zeros((2 * b, 2 * b, 2 * b), dtype='complex')

    for j1 in range(f.shape[0]):
        alpha = 2 * np.pi * j1 / (2. * b)
        for k in range(f.shape[1]):
            beta = np.pi * (2 * k + 1) / (4. * b)
            for j2 in range(f.shape[2]):
                gamma = 2 * np.pi * j2 / (2. * b)
                f[j1, k, j2] = D(alpha, beta, gamma)
    return f

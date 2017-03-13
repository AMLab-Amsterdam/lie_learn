
import numpy as np
from ..SO3FFT_Naive import SO3_FFT_NaiveReal, SO3_FFT_NaiveComplex
from lie_learn.representations.SO3.pinchon_hoggan.pinchon_hoggan_dense import Jd, rot_mat
from lie_learn.representations.SO3.irrep_bases import change_of_basis_matrix


def test_SO3_FFT_Synthesis_NaiveComplex():
    """
    Testing if the complex Naive SO(3) FFT synthesis works correctly for 1-hot input vectors
    """
    L_max = 3

    f_hat = [np.zeros((2 * ll + 1, 2 * ll + 1)) for ll in range(L_max + 1)]
    fft = SO3_FFT_NaiveComplex(L_max=L_max, L2_normalized=False)
    for l in range(L_max + 1):
        for m in range(-l, l + 1):
            for n in range(-l, l + 1):
                f_hat[l][l + m, l + n] = 1.
                D = fft.synthesize(f_hat)
                f_hat[l][l + m, l + n] = 0.
                D2 = make_D_sample_grid(b=L_max + 1, l=l, m=m, n=n,
                                        field='complex', normalization='seismology',
                                        order='centered', condon_shortley='cs')

                diff = np.sum(np.abs(D - D2))
                print(l, m, n, diff)
                # assert np.isclose(diff, 0.0)

                f_hat_2 = fft.analyze(D2)
                # for ff in f_hat_2:
                #    print(np.round(ff, 2))
                f_hat_flat = np.hstack([ff.flatten() for ff in f_hat])
                f_hat_2_flat = np.hstack([ff.flatten() for ff in f_hat_2])

                #print(f_hat_flat)
                #print(f_hat_2_flat)
                #print('f_hat shapes', [ff.shape for ff in f_hat])
                #print('f_hat_2 shapes', [ff.shape for ff in f_hat_2])
                #print('f_hat_flat shape', f_hat_flat.shape)
                #print('f_hat_2_flat shape', f_hat_2_flat.shape)
                #print('D shape', D.shape)
                #print('D2 shape', D2.shape)

                diff = np.sum(np.abs(f_hat_flat - f_hat_2_flat))
                print(l, m, n, diff, np.max(np.abs(f_hat_2_flat)))
                # assert np.isclose(diff, 0.0)


def test_SO3_FFT_Analysis_NaiveComplex():
    """

    """
    L_max = 3

    f_hat = [np.zeros((2 * ll + 1, 2 * ll + 1)) for ll in range(L_max + 1)]
    fft = SO3_FFT_NaiveComplex(L_max=L_max, L2_normalized=False)
    for l in range(L_max + 1):
        for m in range(-l, l + 1):
            for n in range(-l, l + 1):
                f_hat[l][l + m, l + n] = 1.
                D = make_D_sample_grid(b=L_max + 1, l=l, m=m, n=n,
                    field='complex', normalization='seismology', order='centered', condon_shortley='cs')



def checkTODO_SO3_FFT_NaiveComplex_invertible():
    L_max = 3

    f_hat = [np.zeros((2 * ll + 1, 2 * ll + 1)) for ll in range(L_max + 1)]
    fft = SO3_FFT_NaiveComplex(L_max=L_max, L2_normalized=False)
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
    """if D == 'c':
        C2R = change_of_basis_matrix(l,
                                     frm=('complex', 'seismology', 'centered', 'cs'),
                                     to=('real', 'quantum', 'centered', 'cs'))

        D = lambda a, b, c: C2R.conj().T.dot(rot_mat(a, b, c, l, Jd[l])).dot(C2R)[m + l, n + l]

    elif D == 'r':
        D = lambda a, b, c: rot_mat(a, b, c, l, Jd[l])[m + l, n + l]
    else:
        assert False

    B = change_of_basis_matrix(l,
                               frm=(field, normalization, order, condon_shortley),
                               to=('real', 'quantum', 'centered', 'cs'))
    D = lambda a, b, c: B.conj().T.dot(rot_mat(a, b, c, l, Jd[l])).dot(B)[m + l, n + l]
    """

    from lie_learn.representations.SO3.wigner_d import wigner_D_matrix
    D = lambda a, b, c: wigner_D_matrix(l, alpha, beta, gamma,
                                        field=field, normalization=normalization,
                                        order=order, condon_shortley=condon_shortley)[m + l, n + l]

    f = np.zeros((2 * b, 2 * b, 2 * b), dtype='complex')

    # Normalization constant for L2-normalized Wigner D functions
    # Z = (1. / (2 * np.pi)) * np.sqrt(l + 0.5)
    # Z = 1. / np.pi

    # Normalization constant for Haar-normalized Wigner-D functions, as used by S3.integrate()
    # Z = np.sqrt(2 * l + 1)

    Z = 1.

    for j1 in range(f.shape[0]):
        alpha = 2 * np.pi * j1 / (2. * b)
        for k in range(f.shape[1]):
            beta = np.pi * (2 * k + 1) / (4. * b)
            for j2 in range(f.shape[2]):
                gamma = 2 * np.pi * j2 / (2. * b)
                f[j1, k, j2] = D(alpha, beta, gamma) * Z
    return f

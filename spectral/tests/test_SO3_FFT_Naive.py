
import numpy as np
from ..SO3FFT_Naive import SO3_FFT_NaiveReal, SO3_FFT_NaiveComplex
from lie_learn.representations.SO3.pinchon_hoggan.pinchon_hoggan_dense import Jd, rot_mat
from lie_learn.representations.SO3.irrep_bases import change_of_basis_matrix


def test_SO3_FFT_NaiveComplex():
    """
    Testing if the complex Naive SO(3) FFT synthesis works correctly
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
                D2 = make_D_sample_grid(Jd, b=L_max + 1, l=l, m=m, n=n, D='c')

                print(l, m, n, np.sum(np.abs(D - D2)))  #, D2 / D
                assert np.isclose(np.sum(np.abs(D - D2)), 0.0)


def test_SO3_FFT_NaiveReal():
    """
    Testing if the real Naive SO(3) FFT synthesis works correctly
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
                D2 = make_D_sample_grid(Jd, b=L_max + 1, l=l, m=m, n=n, D='r')

                print(l, m, n, np.sum(np.abs(D - D2)))  #, D2.real / D, D, D2
                assert np.isclose(np.sum(np.abs(D - D2)), 0.0)


def make_D_sample_grid(Jd, b=4, l=0, m=0, n=0, D='c'):
    if D == 'c':
        C2R = change_of_basis_matrix(l,
                                     frm=('complex', 'seismology', 'centered', 'cs'),
                                     to=('real', 'quantum', 'centered', 'cs'))

        D = lambda a, b, c: C2R.conj().T.dot(rot_mat(a, b, c, l, Jd[l])).dot(C2R)[m + l, n + l]

    elif D == 'r':
        D = lambda a, b, c: rot_mat(a, b, c, l, Jd[l])[m + l, n + l]
    else:
        assert False

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
                #f[j1, k, j2] = D(alpha, beta, gamma) * (1. / 2. * np.pi) * np.sqrt(l + 0.5)
                #f[j1, k, j2] = (C2R.conj().T.dot(D(alpha, beta, gamma)).dot(C2R))[m + l, n + l]
                #f[j1, k, j2] *= (1. / (2. * np.pi)) * np.sqrt(l + 0.5)

                f[j1, k, j2] = D(alpha, beta, gamma) * Z

    return f




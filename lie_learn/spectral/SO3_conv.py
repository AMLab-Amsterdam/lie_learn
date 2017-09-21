


import numpy as np

from lie_learn.spectral.SO3FFT_Naive import SO3_FT_Naive


def conv_test():
    """
    Compute the convolution of two functions on SO(3).
    Let f1 : SO(3) -> R and f2 : SO(3) -> R, then the convolution is defined as
    f1 * f2(g) = int_{SO(3)} f1(h) f2(g^{-1} h) dh,
    where g in SO(3) and dh is the normalized Haar measure on SO(3).

    The convolution is computed by a Fourier transform.
    It can be shown that the SO(3) Fourier transform of the convolution f1 * f2 is equal to the matrix product
    of the SO(3) Fourier transforms of f1 and f2.
    For more details, see the note on "Convolution on S^2 and SO(3)"

    :return:
    """
    from lie_learn.spectral.SO3FFT_Naive import SO3_FT_Naive

    b = 10
    f1 = np.ones((2 * b + 2, b + 1)) #TODO
    f2 = np.ones((2 * b + 2, b + 1))

    s2_fft = S2_FT_Naive(L_max=b - 1, grid_type='Gauss-Legendre',
                         field='real', normalization='quantum', condon_shortley='cs')

    so3_fft = SO3_FT_Naive(L_max=b - 1,
                           field='real', normalization='quantum', order='centered', condon_shortley='cs')

    # Spherical Fourier transform
    f1_hat = s2_fft.analyze(f1)
    f2_hat = s2_fft.analyze(f2)

    # Perform block-wise outer product
    f12_hat = []
    for l in range(b):
        f1_hat_l = f1_hat[l ** 2:l ** 2 + 2 * l + 1]
        f2_hat_l = f2_hat[l ** 2:l ** 2 + 2 * l + 1]

        f12_hat_l = f1_hat_l[:, None] * f2_hat_l[None, :].conj()
        f12_hat.append(f12_hat_l)

    # Inverse SO(3) Fourier transform
    f12 = so3_fft.synthesize(f12_hat)

    return f12


def SO3_convolve(f, g, dw=None, d=None):

    assert f.shape == g.shape
    assert f.shape[0] % 2 == 0
    b = f.shape[0] / 2

    if d is None:
        d = setup_d_transform(b)

    # To convolve, first perform a Fourier transform on f and g:
    F = SO3_fft(f, dw)
    G = SO3_fft(g, dw)

    # The Fourier transform of the convolution f*g is the matrix product FG
    # of their Fourier transforms F and G:
    FG = [np.dot(a, b) for (a, b) in zip(F, G)]

    # The convolution is obtain by inverse Fourier transforming FG:
    return SO3_ifft(FG, d)
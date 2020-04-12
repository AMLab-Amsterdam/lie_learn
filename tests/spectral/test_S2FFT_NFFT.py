import lie_learn.spaces.S2 as S2
from lie_learn.spectral.S2FFT_NFFT import S2FFT_NFFT
from lie_learn.representations.SO3.spherical_harmonics import *


def test_S2FFT_NFFT():
    """
    Testing S2FFT NFFT
    """
    b = 8
    convention = 'Gauss-Legendre'
    #convention = 'Clenshaw-Curtis'
    x = S2.meshgrid(b=b, grid_type=convention)
    print(x[0].shape, x[1].shape)
    x = np.c_[x[0][..., None], x[1][..., None]]#.reshape(-1, 2)
    print(x.shape)
    x = x.reshape(-1, 2)
    w = S2.quadrature_weights(b=b, grid_type=convention).flatten()
    F = S2FFT_NFFT(L_max=b, x=x, w=w)

    for l in range(0, b):
        for m in range(-l, l + 1):
            #l = b; m = b
            f = sh(l, m, x[..., 0], x[..., 1], field='real', normalization='quantum', condon_shortley=True)
            #f2 = np.random.randn(*f.shape)
            print(f)

            f_hat = F.analyze(f)
            print(np.round(f_hat, 3))
            f_reconst = F.synthesize(f_hat)

            #print np.round(f, 3)
            print(np.round(f_reconst, 3))
            #print np.round(f/f_reconst, 3)
            print(np.abs(f-f_reconst).sum())
            assert np.isclose(np.abs(f-f_reconst).sum(), 0.)

            print(np.round(f_hat, 3))
            assert np.isclose(f_hat[l ** 2 + l + m], 1.)
            #assert False
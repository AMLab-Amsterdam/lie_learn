"""
The 2-sphere, S^2
"""
import numpy as np
from numpy.polynomial.legendre import leggauss


def change_coordinates(coords, p_from='C', p_to='S'):
    """
    Change Spherical to Cartesian coordinates and vice versa, for points x in S^2.

    In the spherical system, we have coordinates beta and alpha,
    where beta in [0, pi] and alpha in [0, 2pi]
    
    We use the names beta and alpha for compatibility with the SO(3) code (S^2 being a quotient SO(3)/SO(2)).
    Many sources, like wikipedia use theta=beta and phi=alpha.

    :param coords: coordinate array
    :param p_from: 'C' for Cartesian or 'S' for spherical coordinates
    :param p_to: 'C' for Cartesian or 'S' for spherical coordinates
    :return: new coordinates
    """
    if p_from == p_to:
        return coords
    elif p_from == 'S' and p_to == 'C':

        beta = coords[..., 0]
        alpha = coords[..., 1]
        r = 1.

        out = np.empty(beta.shape + (3,))

        ct = np.cos(beta)
        cp = np.cos(alpha)
        st = np.sin(beta)
        sp = np.sin(alpha)
        out[..., 0] = r * st * cp  # x
        out[..., 1] = r * st * sp  # y
        out[..., 2] = r * ct       # z
        return out

    elif p_from == 'C' and p_to == 'S':

        x = coords[..., 0]
        y = coords[..., 1]
        z = coords[..., 2]

        out = np.empty(x.shape + (2,))
        out[..., 0] = np.arccos(z)         # beta
        out[..., 1] = np.arctan2(y, x)     # alpha
        return out

    else:
        raise ValueError('Unknown conversion:' + str(p_from) + ' to ' + str(p_to))


def meshgrid(b, grid_type='Driscoll-Healy'):
    """
    Create a coordinate grid for the 2-sphere.
    There are various ways to setup a grid on the sphere.

    if grid_type == 'Driscoll-Healy', we follow the grid_type from [4], which is also used in [5]:
    beta_j = pi j / (2 b)     for j = 0, ..., 2b - 1
    alpha_k = pi k / b           for k = 0, ..., 2b - 1

    if grid_type == 'SOFT', we follow the grid_type from [1] and [6]
    beta_j = pi (2 j + 1) / (4 b)   for j = 0, ..., 2b - 1
    alpha_k = pi k / b                for k = 0, ..., 2b - 1

    if grid_type == 'Clenshaw-Curtis', we use the Clenshaw-Curtis grid, as defined in [2] (section 6):
    beta_j = j pi / (2b)     for j = 0, ..., 2b
    alpha_k = k pi / (b + 1)    for k = 0, ..., 2b + 1

    if grid_type == 'Gauss-Legendre', we use the Gauss-Legendre grid, as defined in [2] (section 6) and [7] (eq. 2):
    beta_j = the Gauss-Legendre nodes    for j = 0, ..., b
    alpha_k = k pi / (b + 1),               for k = 0, ..., 2 b + 1

    if grid_type == 'HEALPix', we use the HEALPix grid, see [2] (section 6):
    TODO

    if grid_type == 'equidistribution', we use the equidistribution grid, as defined in [2] (section 6):
    TODO

    [1] SOFT: SO(3) Fourier Transforms
    Kostelec, Peter J & Rockmore, Daniel N.

    [2] Fast evaluation of quadrature formulae on the sphere
    Jens Keiner, Daniel Potts

    [3] A Fast Algorithm for Spherical Grid Rotations and its Application to Singular Quadrature
    Zydrunas Gimbutas Shravan Veerapaneni

    [4] Computing Fourier transforms and convolutions on the 2-sphere
    Driscoll, JR & Healy, DM

    [5] Engineering Applications of Noncommutative Harmonic Analysis
    Chrikjian, G.S. & Kyatkin, A.B.

    [6] FFTs for the 2-Sphere – Improvements and Variations
    Healy, D., Rockmore, D., Kostelec, P., Moore, S

    [7] A Fast Algorithm for Spherical Grid Rotations and its Application to Singular Quadrature
    Zydrunas Gimbutas, Shravan Veerapaneni

    :param b: the bandwidth / resolution
    :return: a meshgrid on S^2
    """
    return np.meshgrid(*linspace(b, grid_type), indexing='ij')


def linspace(b, grid_type='Driscoll-Healy'):
    if grid_type == 'Driscoll-Healy':
        beta = np.arange(2 * b) * np.pi / (2. * b)
        alpha = np.arange(2 * b) * np.pi / b
    elif grid_type == 'SOFT':
        beta = np.pi * (2 * np.arange(2 * b) + 1) / (4. * b)
        alpha = np.arange(2 * b) * np.pi / b
    elif grid_type == 'Clenshaw-Curtis':
        # beta = np.arange(2 * b + 1) * np.pi / (2 * b)
        # alpha = np.arange(2 * b + 2) * np.pi / (b + 1)
        # Must use np.linspace to prevent numerical errors that cause beta > pi
        beta = np.linspace(0, np.pi, 2 * b + 1)
        alpha = np.linspace(0, 2 * np.pi, 2 * b + 2, endpoint=False)
    elif grid_type == 'Gauss-Legendre':
        x, _ = leggauss(b + 1)  # TODO: leggauss docs state that this may not be only stable for orders > 100
        beta = np.arccos(x)
        alpha = np.arange(2 * b + 2) * np.pi / (b + 1)
    elif grid_type == 'HEALPix':
        #TODO: implement this here so that we don't need the dependency on healpy / healpix_compat
        from healpix_compat import healpy_sphere_meshgrid
        return healpy_sphere_meshgrid(b)
    elif grid_type == 'equidistribution':
        raise NotImplementedError('Not implemented yet; see Fast evaluation of quadrature formulae on the sphere.')
    else:
        raise ValueError('Unknown grid_type:' + grid_type)
    return beta, alpha


def quadrature_weights(b, grid_type='Gauss-Legendre'):
    """
    Compute quadrature weights for a given grid-type.
    The function S2.meshgrid generates the points that correspond to the weights generated by this function.

    if convention == 'Gauss-Legendre':
    The quadrature formula is exact for polynomials up to degree M less than or equal to 2b + 1,
    so that we can compute exact Fourier coefficients for f a polynomial of degree at most b.

    if convention == 'Clenshaw-Curtis':
    The quadrature formula is exact for polynomials up to degree M less than or equal to 2b,
    so that we can compute exact Fourier coefficients for f a polynomial of degree at most b.

    :param b: the grid resolution. See S2.meshgrid
    :param grid_type:
    :return:
    """
    if grid_type == 'Clenshaw-Curtis':
        # There is a faster fft based method to compute these weights
        # see "Fast evaluation of quadrature formulae on the sphere"
        # W = np.empty((2 * b + 2, 2 * b + 1))
        # for j in range(2 * b + 1):
        #    eps_j_2b = 0.5 if j == 0 or j == 2 * b else 1.
        #    for k in range(2 * b + 2):  # Doesn't seem to depend on k..
        #        W[k, j] = (4 * np.pi * eps_j_2b) / (b * (2 * b + 2))
        #        sum = 0.
        #        for l in range(b + 1):
        #            eps_l_b = 0.5 if l == 0 or l == b else 1.
        #            sum += eps_l_b / (1 - 4 * l ** 2) * np.cos(j * l * np.pi / b)
        #        W[k, j] *= sum
        w = _clenshaw_curtis_weights(n=2 * b)
        W = np.empty((2 * b + 1, 2 * b + 2))
        W[:] = w[:, None]
    elif grid_type == 'Gauss-Legendre':
        # We found this formula in:
        # "A Fast Algorithm for Spherical Grid Rotations and its Application to Singular Quadrature"
        # eq. 10
        _, w = leggauss(b + 1)
        W = w[:, None] * (2 * np.pi / (2 * b + 2) * np.ones(2 * b + 2)[None, :])
    elif grid_type == 'SOFT':
        print("WARNING: SOFT quadrature weights don't work yet")
        k = np.arange(0, b)
        w = np.array([(2. / b) * np.sin(np.pi * (2. * j + 1.) / (4. * b)) *
                      (np.sum((1. / (2 * k + 1))
                              * np.sin((2 * j + 1) * (2 * k + 1)
                                       * np.pi / (4. * b))))
                      for j in range(2 * b)])
        W = w[:, None] * np.ones(2 * b)[None, :]
    else:
        raise ValueError('Unknown grid_type:' + str(grid_type))

    return W


def integrate(f, normalize=True):
    """
    Integrate a function f : S^2 -> R over the sphere S^2, using the invariant integration measure
    mu((beta, alpha)) = sin(beta) dbeta dalpha
    i.e. this returns
    int_S^2 f(x) dmu(x) = int_0^2pi int_0^pi f(beta, alpha) sin(beta) dbeta dalpha

    :param f: a function of two scalar variables returning a scalar.
    :return: the integral of f over the 2-sphere
    """
    from scipy.integrate import quad

    f2 = lambda alpha: quad(lambda beta: f(beta, alpha) * np.sin(beta),
                          a=0,
                          b=np.pi)[0]
    integral = quad(f2, 0, 2 * np.pi)[0]

    if normalize:
        return integral / (4 * np.pi)
    else:
        return integral


def integrate_quad(f, grid_type, normalize=True, w=None):
    """
    Integrate a function f : S^2 -> R, sampled on a grid of type grid_type, using quadrature weights w.

    :param f: an ndarray containing function values on a grid
    :param grid_type: the type of grid used to sample f
    :param normalize: whether to use the normalized Haar measure or not
    :param w: the quadrature weights. If not given, they are computed.
    :return: the integral of f over S^2.
    """

    if grid_type != 'Gauss-Legendre' and grid_type != 'Clenshaw-Curtis':
        raise NotImplementedError

    b = (f.shape[1] - 2) // 2  # This works for Gauss-Legendre and Clenshaw-Curtis

    if w is None:
        w = quadrature_weights(b, grid_type)

    integral = np.sum(f * w)

    if normalize:
        return integral / (4 * np.pi)
    else:
        return integral


def plot_sphere_func(f, grid='Clenshaw-Curtis', beta=None, alpha=None, colormap='jet', fignum=0, normalize=True):

    #TODO: All grids except Clenshaw-Curtis have holes at the poles
    # TODO: update this function now that we changed the order of axes in f

    import matplotlib
    matplotlib.use('WxAgg')
    matplotlib.interactive(True)
    from mayavi import mlab

    if normalize:
        f = (f - np.min(f)) / (np.max(f) - np.min(f))

    if grid == 'Driscoll-Healy':
        b = f.shape[0] / 2
    elif grid == 'Clenshaw-Curtis':
        b = (f.shape[0] - 2) / 2
    elif grid == 'SOFT':
        b = f.shape[0] / 2
    elif grid == 'Gauss-Legendre':
        b = (f.shape[0] - 2) / 2

    if beta is None or alpha is None:
        beta, alpha = meshgrid(b=b, grid_type=grid)

    alpha = np.r_[alpha, alpha[0, :][None, :]]
    beta = np.r_[beta, beta[0, :][None, :]]
    f = np.r_[f, f[0, :][None, :]]

    x = np.sin(beta) * np.cos(alpha)
    y = np.sin(beta) * np.sin(alpha)
    z = np.cos(beta)

    mlab.figure(fignum, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(600, 400))
    mlab.clf()
    mlab.mesh(x, y, z, scalars=f, colormap=colormap)

    #mlab.view(90, 70, 6.2, (-1.3, -2.9, 0.25))
    mlab.show()


def plot_sphere_func2(f, grid='Clenshaw-Curtis', beta=None, alpha=None, colormap='jet', fignum=0,  normalize=True):
    # TODO: update this  function now that we have changed the order of axes in f
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np
    from scipy.special import sph_harm

    if normalize:
        f = (f - np.min(f)) / (np.max(f) - np.min(f))

    if grid == 'Driscoll-Healy':
        b = f.shape[0] // 2
    elif grid == 'Clenshaw-Curtis':
        b = (f.shape[0] - 2) // 2
    elif grid == 'SOFT':
        b = f.shape[0] // 2
    elif grid == 'Gauss-Legendre':
        b = (f.shape[0] - 2) // 2

    if beta is None or alpha is None:
        beta, alpha = meshgrid(b=b, grid_type=grid)

    alpha = np.r_[alpha, alpha[0, :][None, :]]
    beta = np.r_[beta, beta[0, :][None, :]]
    f = np.r_[f, f[0, :][None, :]]

    x = np.sin(beta) * np.cos(alpha)
    y = np.sin(beta) * np.sin(alpha)
    z = np.cos(beta)

    # m, l = 2, 3
    # Calculate the spherical harmonic Y(l,m) and normalize to [0,1]
    # fcolors = sph_harm(m, l, beta, alpha).real
    # fmax, fmin = fcolors.max(), fcolors.min()
    # fcolors = (fcolors - fmin) / (fmax - fmin)
    print(x.shape, f.shape)

    if f.ndim == 2:
        f = cm.gray(f)
        print('2')

    # Set the aspect ratio to 1 so our sphere looks spherical
    fig = plt.figure(figsize=plt.figaspect(1.))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=f ) # cm.gray(f))
    # Turn off the axis planes
    ax.set_axis_off()
    plt.show()


def _clenshaw_curtis_weights(n):
    """
    Computes the Clenshaw-Curtis quadrature using a fast FFT method.

    This is a 'brainless' port of MATLAB code found in:
    Fast Construction of the Fejer and Clenshaw-Curtis Quadrature Rules
    Jorg Waldvogel, 2005
    http://www.sam.math.ethz.ch/~joergw/Papers/fejer.pdf

    :param n:
    :return:
    """
    from scipy.fftpack import ifft, fft, fftshift

    # TODO python3 handles division differently from python2. Check how MATLAB interprets /, and if this code is still correct for python3

    # function [wf1,wf2,wcc] = fejer(n)
    # Weights of the Fejer2, Clenshaw-Curtis and Fejer1 quadratures by DFTs
    # n>1. Nodes: x_k = cos(k*pi/n)
    # N = [1:2:n-1]'; l=length(N); m=n-l; K=[0:m-1]';
    N = np.arange(start=1, stop=n, step=2)[:, None]
    l = N.size
    m = n - l
    K = np.arange(start=0, stop=m)[:, None]

    # Fejer2 nodes: k=0,1,...,n; weights: wf2, wf2_n=wf2_0=0
    # v0 = [2./N./(N-2); 1/N(end); zeros(m,1)];
    v0 = np.vstack([2. / N / (N-2), 1. / N[-1]] + [0] * m)

    # v2 = -v0(1:end-1) - v0(end:-1:2);
    # wf2 = ifft(v2);
    v2 = -v0[:-1] - v0[:0:-1]

    # Clenshaw-Curtis nodes: k=0,1,...,n; weights: wcc, wcc_n=wcc_0
    # g0 = -ones(n,1);
    g0 = -np.ones((n, 1))

    # g0(1 + l) = g0(1 + l) + n;
    g0[l] = g0[l] + n

    # g0(1+m) = g0(1 + m) + n;
    g0[m] = g0[m] + n

    # g = g0/(n^2-1+mod(n,2));
    g = g0 / (n ** 2 - 1 + n % 2)

    # wcc=ifft(v2 + g);
    wcc = ifft((v2 + g).flatten()).real
    wcc = np.hstack([wcc, wcc[0]])

    # Fejer1 nodes: k=1/2,3/2,...,n-1/2; vector of weights: wf1
    # v0=[2*exp(i*pi*K/n)./(1-4*K.^2); zeros(l+1,1)];
    # v1=v0(1:end-1)+conj(v0(end:-1:2)); wf1=ifft(v1);
    # don't need these

    return wcc * np.pi / (n / 2 + 1)  # adjust for different scaling of python vs MATLAB fft

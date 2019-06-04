
import numpy as np
from scipy.special import sph_harm, lpmv
try:
    from scipy.misc import factorial
except:
    from scipy.special import factorial

def sh(l, m, theta, phi, field='real', normalization='quantum', condon_shortley=True):
    if field == 'real':
        return rsh(l, m, theta, phi, normalization, condon_shortley)
    elif field == 'complex':
        return csh(l, m, theta, phi, normalization, condon_shortley)
    else:
        raise ValueError('Unknown field: ' + str(field))


def sh_squared_norm(l, normalization='quantum', normalized_haar=True):
    """
    Compute the squared norm of the spherical harmonics.

    The squared norm of a function on the sphere is defined as
    |f|^2 = int_S^2 |f(x)|^2 dx
    where dx is a Haar measure.

    :param l: for some normalization conventions, the norm of a spherical harmonic Y^l_m depends on the degree l
    :param normalization: normalization convention for the spherical harmonic
    :param normalized_haar: whether to use the Haar measure da db sinb or the normalized Haar measure da db sinb / 4pi
    :return: the squared norm of the spherical harmonic with respect to given measure
    """
    if normalization == 'quantum' or normalization == 'seismology':
        # The quantum and seismology spherical harmonics are normalized with respect to the Haar measure
        # dmu(theta, phi) = dtheta sin(theta) dphi
        sqnorm = 1.
    elif normalization == 'geodesy':
        # The geodesy spherical harmonics are normalized with respect to the *normalized* Haar measure
        # dmu(theta, phi) = dtheta sin(theta) dphi / 4pi
        sqnorm = 4 * np.pi
    elif normalization == 'nfft':
        sqnorm = 4 * np.pi / (2 * l + 1)
    else:
        raise ValueError('Unknown normalization')

    if normalized_haar:
        return sqnorm / (4 * np.pi)
    else:
        return sqnorm


def block_sh_ph(L_max, theta, phi):
    """
    Compute all spherical harmonics up to and including degree L_max, for angles theta and phi.

    This function is currently rather hacky, but the method used here is very fast and stable, compared
    to builtin scipy functions.

    :param L_max:
    :param theta:
    :param phi:
    :return:
    """

    from .pinchon_hoggan.pinchon_hoggan import apply_rotation_block, make_c2b
    from .irrep_bases import change_of_basis_function

    irreps = np.arange(L_max + 1)

    ls = [[ls] * (2 * ls + 1) for ls in irreps]
    ls = np.array([ll for sublist in ls for ll in sublist])  # 0, 1, 1, 1, 2, 2, 2, 2, 2, ...
    ms = [list(range(-ls, ls + 1)) for ls in irreps]
    ms = np.array([mm for sublist in ms for mm in sublist])  # 0, -1, 0, 1, -2, -1, 0, 1, 2, ...

    # Get a vector Y that selects the 0-frequency component from each irrep in the centered basis
    # If D is a Wigner D matrix, then D Y is the center column of D, which is equal to the spherical harmonics.
    Y = (ms == 0).astype(float)

    # Change to / from the block basis (since the rotation code works in that basis)
    c2b = change_of_basis_function(irreps,
                                   frm=('real', 'quantum', 'centered', 'cs'),
                                   to=('real', 'quantum', 'block', 'cs'))
    b2c = change_of_basis_function(irreps,
                                   frm=('real', 'quantum', 'block', 'cs'),
                                   to=('real', 'quantum', 'centered', 'cs'))

    Yb = c2b(Y)

    # Rotate Yb:
    c2b = make_c2b(irreps)
    import os
    J_block = np.load(os.path.join(os.path.dirname(__file__), 'pinchon_hoggan', 'J_block_0-278.npy'), allow_pickle=True)
    J_block = list(J_block[irreps])

    g = np.zeros((theta.size, 3))
    g[:, 0] = phi
    g[:, 1] = theta
    TYb = apply_rotation_block(g=g, X=Yb[np.newaxis, :],
                               irreps=irreps, c2b=c2b,
                               J_block=J_block, l_max=np.max(irreps))

    print(Yb.shape, TYb.shape)

    # Change back to centered basis
    TYc = b2c(TYb.T).T  # b2c doesn't work properly for matrices, so do a transpose hack

    print(TYc.shape)

    # Somehow, the SH obtained so far are equal to real, nfft, cs spherical harmonics
    # Change to real quantum centered cs
    c = change_of_basis_function(irreps,
                                 frm=('real', 'nfft', 'centered', 'cs'),
                                 to=('real', 'quantum', 'centered', 'cs'))
    TYc2 = c(TYc)
    print(TYc2.shape)

    return TYc2


def rsh(l, m, theta, phi, normalization='quantum', condon_shortley=True):
    """
    Compute the real spherical harmonic (RSH) S_l^m(theta, phi).

    The RSH are obtained from Complex Spherical Harmonics (CSH) as follows:
    if m < 0:
        S_l^m = i / sqrt(2) * (Y_l^m - (-1)^m Y_l^{-m})
    if m == 0:
        S_l^m = Y_l^0
    if m > 0:
        S_l^m = 1 / sqrt(2) * (Y_l^{-m} + (-1)^m Y_l^m)
     (see [1])

    Various normalizations for the CSH exist, see the CSH() function. Since the CSH->RSH change of basis is unitary,
    the orthogonality and normalization properties of the RSH are the same as those of the CSH from which they were
    obtained. Furthermore, the operation of changing normalization and that of changeing field
    (complex->real or vice-versa) commute, because the ratio c_m of normalization constants are always the same for
    m and -m (to see this that this implies commutativity, substitute Y_l^m * c_m for Y_l^m in the above formula).

    Pinchon & Hoggan [2] define a different change of basis for CSH -> RSH, but they also use an unusual definition
    of CSH. To obtain RSH as defined by Pinchon-Hoggan, use this function with normalization='quantum'.

    References:
    [1] http://en.wikipedia.org/wiki/Spherical_harmonics#Real_form
    [2] Rotation matrices for real spherical harmonics: general rotations of atomic orbitals in space-fixed axes.

    :param l: non-negative integer; the degree of the CSH.
    :param m: integer, -l <= m <= l; the order of the CSH.
    :param theta: the colatitude / polar angle,
    ranging from 0 (North Pole, (X,Y,Z)=(0,0,1)) to pi (South Pole, (X,Y,Z)=(0,0,-1)).
    :param phi: the longitude / azimuthal angle, ranging from 0 to 2 pi.
    :param normalization: how to normalize the RSH:
    'seismology', 'quantum', 'geodesy'.
    these are immediately passed to the CSH functions, and since the change of basis
    from CSH to RSH is unitary, the orthogonality and normalization properties are unchanged.
    :return: the value of the real spherical harmonic S^l_m(theta, phi)
    """
    l, m, theta, phi = np.broadcast_arrays(l, m, theta, phi)
    # Get the CSH for m and -m, using Condon-Shortley phase (regardless of whhether CS is requested or not)
    # The reason is that the code that changes from CSH to RSH assumes CS phase.

    a = csh(l=l, m=m, theta=theta, phi=phi, normalization=normalization, condon_shortley=True)
    b = csh(l=l, m=-m, theta=theta, phi=phi, normalization=normalization, condon_shortley=True)

    #if m > 0:
    #    y = np.array((b + ((-1.)**m) * a).real / np.sqrt(2.))
    #elif m < 0:
    #    y = np.array((1j * a - 1j * ((-1.)**(-m)) * b).real / np.sqrt(2.))
    #else:
    #    # For m == 0, the complex spherical harmonics are already real
    #    y = np.array(a.real)

    y = ((m > 0) * np.array((b + ((-1.)**m) * a).real / np.sqrt(2.))
         + (m < 0) * np.array((1j * a - 1j * ((-1.)**(-m)) * b).real / np.sqrt(2.))
         + (m == 0) * np.array(a.real))

    if condon_shortley:
        return y
    else:
        # Cancel the CS phase of y (i.e. multiply by -1 when m is both odd and greater than 0)
        return y * ((-1.) ** (m * (m > 0)))


def csh(l, m, theta, phi, normalization='quantum', condon_shortley=True):
    """
    Compute Complex Spherical Harmonics (CSH) Y_l^m(theta, phi).
    Unlike the scipy.special.sph_harm function, we use the common convention that
    theta is the polar angle (0 to pi) and phi is the azimuthal angle (0 to 2pi).

    The spherical harmonic 'backbone' is:
    Y_l^m(theta, phi) = P_l^m(cos(theta)) exp(i m phi)
    where P_l^m is the associated Legendre function as defined in the scipy library (scipy.special.sph_harm).

    Various normalization factors can be multiplied with this function.
    -> seismology: sqrt( ((2 l + 1) * (l - m)!) / (4 pi * (l + m)!) )
    -> quantum: (-1)^2 sqrt( ((2 l + 1) * (l - m)!) / (4 pi * (l + m)!) )
    -> unnormalized: 1
    -> geodesy: sqrt( ((2 l + 1) * (l - m)!) / (l + m)! )
    -> nfft: sqrt( (l - m)! / (l + m)! )

    The 'quantum' and 'seismology' CSH are normalized so that
    <Y_l^m, Y_l'^m'>
    =
    int_S^2 Y_l^m(theta, phi) Y_l'^m'* dOmega
    =
    delta(l, l') delta(m, m')
    where dOmega is the volume element for the sphere S^2:
    dOmega = sin(theta) dtheta dphi
    The 'geodesy' convention have unit power, meaning the norm is equal to the surface area of the unit sphere (4 pi)
    <Y_l^m, Y_l'^m'> = 4pi delta(l, l') delta(m, m')
    So these are orthonormal with respect to the *normalized* Haar measure sin(theta) dtheta dphi / 4pi

    On each of these normalizations, one can optionally include a Condon-Shortley phase factor:
    (-1)^m   (if m > 0)
    1        (otherwise)
    Note that this is the definition of Condon-Shortley according to wikipedia [1], but other sources call a
    phase factor of (-1)^m a Condon-Shortley phase (without mentioning the condition m > 0).

    References:
    [1] http://en.wikipedia.org/wiki/Spherical_harmonics#Conventions

    :param l: non-negative integer; the degree of the CSH.
    :param m: integer, -l <= m <= l; the order of the CSH.
    :param theta: the colatitude / polar angle,
    ranging from 0 (North Pole, (X,Y,Z)=(0,0,1)) to pi (South Pole, (X,Y,Z)=(0,0,-1)).
    :param phi: the longitude / azimuthal angle, ranging from 0 to 2 pi.
    :param normalization: how to normalize the CSH:
    'seismology', 'quantum', 'geodesy', 'unnormalized', 'nfft'.
    :return: the value of the complex spherical harmonic Y^l_m(theta, phi)
    """
    # NOTE: it seems like in the current version of scipy.special, sph_harm no longer accepts keyword arguments,
    # so I'm removing them. I hope the order of args hasn't changed
    if normalization == 'quantum':
        # y = ((-1.) ** m) * sph_harm(m, l, theta=phi, phi=theta)
        y = ((-1.) ** m) * sph_harm(m, l, phi, theta)
    elif normalization == 'seismology':
        # y = sph_harm(m, l, theta=phi, phi=theta)
        y = sph_harm(m, l, phi, theta)
    elif normalization == 'geodesy':
        # y = np.sqrt(4 * np.pi) * sph_harm(m, l, theta=phi, phi=theta)
        y = np.sqrt(4 * np.pi) * sph_harm(m, l, phi, theta)
    elif normalization == 'unnormalized':
        # y = sph_harm(m, l, theta=phi, phi=theta) / np.sqrt((2 * l + 1) * factorial(l - m) /
        #                                                    (4 * np.pi * factorial(l + m)))
        y = sph_harm(m, l, phi, theta) / np.sqrt((2 * l + 1) * factorial(l - m) /
                                                 (4 * np.pi * factorial(l + m)))
    elif normalization == 'nfft':
        # y = sph_harm(m, l, theta=phi, phi=theta) / np.sqrt((2 * l + 1) / (4 * np.pi))
        y = sph_harm(m, l, phi, theta) / np.sqrt((2 * l + 1) / (4 * np.pi))
    else:
        raise ValueError('Unknown normalization convention:' + str(normalization))

    if condon_shortley:
        # The sph_harm function already includes CS phase
        return y
    else:
        # Cancel the CS phase in sph_harm (i.e. multiply by -1 when m is both odd and greater than 0)
        return y * ((-1.) ** (m * (m > 0)))


# For testing only:
def _naive_csh_unnormalized(l, m, theta, phi):
    """
    Compute unnormalized SH
    """
    return lpmv(m, l, np.cos(theta)) * np.exp(1j * m * phi)


def _naive_csh_quantum(l, m, theta, phi):
    """
    Compute orthonormalized spherical harmonics in a naive way.
    """
    return (((-1.) ** m) * lpmv(m, l, np.cos(theta)) * np.exp(1j * m * phi) *
            np.sqrt(((2 * l + 1) * factorial(l - m))
                    /
                    (4 * np.pi * factorial(l + m))))


def _naive_csh_seismology(l, m, theta, phi):
    """
    Compute the spherical harmonics according to the seismology convention, in a naive way.
    This appears to be equal to the sph_harm function in scipy.special.
    """
    return (lpmv(m, l, np.cos(theta)) * np.exp(1j * m * phi) *
            np.sqrt(((2 * l + 1) * factorial(l - m))
                    /
                    (4 * np.pi * factorial(l + m))))


def _naive_csh_ph(l, m, theta, phi):
    """
    CSH as defined by Pinchon-Hoggan. Same as wikipedia's quantum-normalized SH = naive_Y_quantum()
    """
    if l == 0 and m == 0:
        return 1. / np.sqrt(4 * np.pi)
    else:
        phase = ((1j) ** (m + np.abs(m)))
        normalizer = np.sqrt(((2 * l + 1.) * factorial(l - np.abs(m)))
                             /
                             (4 * np.pi * factorial(l + np.abs(m))))
        P = lpmv(np.abs(m), l, np.cos(theta))
        e = np.exp(1j * m * phi)
        return phase * normalizer * P * e


def _naive_rsh_ph(l, m, theta, phi):

    if m == 0:
        return np.sqrt((2 * l + 1.) / (4 * np.pi)) * lpmv(m, l, np.cos(theta))
    elif m < 0:
        return np.sqrt(((2 * l + 1.) * factorial(l + m)) /
                       (2 * np.pi * factorial(l - m))) * lpmv(-m, l, np.cos(theta)) * np.sin(-m * phi)
    elif m > 0:
        return np.sqrt(((2 * l + 1.) * factorial(l - m)) /
                       (2 * np.pi * factorial(l + m))) * lpmv(m, l, np.cos(theta)) * np.cos(m * phi)

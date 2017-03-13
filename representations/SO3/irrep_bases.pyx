"""
There are a number of different bases for the irreducible representations of SO(3),
each of which results in a different form for the irrep matrices.
This file contains routines that produce change-of-basis matrices
to take you from one basis to the others.

Recall that all irreducible representations of SO(3) appear in the
decomposition of the regular representations on well-behaved functions
f: S^2 -> C or f : S^2 -> R
from the sphere S^2 to the real or complex numbers.

The regular representation is defined by left translation:
(T(g) f)(h) = f(g^{-1} h)

The most common basis for the irreducible representation of weight l are some
form of *complex* spherical harmonics (CSH) Y_l^m, for -l <= m <= l.

For real functions, one can use real spherical harmonics (RSH) S_l^m,
which have the same indexing scheme and are related to the CSH
by a unitary change of basis.

For both CSH and RSH, there are a number of normalization conventions,
as described in spherical_harmonics.py and in [1]. However, these differ
by either
1) a constant scale factor of sqrt(4 pi), or
2) a scale factor (-1)^m, which is the same for +m and -m.
Since the RSH S_l^m is obtained by a linear combination of complex Y_l^m and Y_l^{-m} (see [1]),
the process of changing normalization and that of changing CSH to RSH commute (we can pull out the scale/phase factor).
Since the CSH-RSH change of basis is a unitary transformation, the change of basis maps each kind of CSH to a kind of
 RSH that has the same normalization properties.

When changing the normalization, the change-of-basis matrix need not be unitary.
In particular, all changes in normalization, except quantum <--> seismology, lead to non-unitary matrices.

Besides normalization, the harmonics can be rearanged in different orders than m=-l,...,l
This is useful because the Pinchon-Hoggan J matrix assumes a block structure in a certain ordering.

For each normalization convention, we have the following bases:
- Complex centered (cc): Y^{-l}, ..., Y^{l}
- Real centered (rc): S^{-l}, ..., S^{l}
- Real block Pinchon-Hoggan (rb): this basis is aligned with the subspaces
  E_xyz,k (etc.) described by Pinchon & Hoggan, and is obtained by a reordering of the RSH.
  In this basis, the Pinchon-Hoggan J matrix has a block structure.

References:
[1] http://en.wikipedia.org/wiki/Spherical_harmonics#Conventions
[2] Rotation matrices for real spherical harmonics: general rotations of atomic orbitals in space-fixed axes.
"""

import numpy as np
cimport numpy as np
import collections
from scipy.linalg import block_diag

INT_TYPE = np.int64
ctypedef np.int64_t INT_TYPE_t

FLOAT_TYPE = np.float64
ctypedef np.float64_t FLOAT_TYPE_t

COMPLEX_TYPE = np.complex128
ctypedef np.complex128_t COMPLEX_TYPE_t


def change_of_basis_matrix(l, frm=('complex', 'seismology', 'centered', 'cs'), to=('real', 'quantum', 'centered', 'cs')):
    """
    Compute change-of-basis matrix that takes the 'frm' basis to the 'to' basis.
    Each basis is identified by:
     1) A field (real or complex)
     2) A normalization / phase convention ('seismology', 'quantum', 'nfft', or 'geodesy')
     3) An ordering convention ('centered', 'block')
     4) Whether to use Condon-Shortley phase (-1)^m for m > 0 ('cs', 'nocs')

    Let B = change_of_basis_matrix(l, frm, to).
    Then if Y is a vector in the frm basis, B.dot(Y) represents the same vector in the to basis.

    :param l: the weight (non-negative integer) of the irreducible representation, or an iterable of weights.
    :param frm: a 3-tuple (field, normalization, ordering) indicating the input basis.
    :param to: a 3-tuple (field, normalization, ordering) indicating the output basis.
    :return: a (2 * l + 1, 2 * l + 1) change of basis matrix.
    """
    from_field, from_normalization, from_ordering, from_cs = frm
    to_field, to_normalization, to_ordering, to_cs = to

    if isinstance(l, collections.Iterable):
        blocks = [change_of_basis_matrix(li, frm, to)
                  for li in l]
        return block_diag(*blocks)

    # First, bring us to the centered basis:
    if from_ordering == 'block':
        B = _c2b(l).T
    elif from_ordering == 'centered':
        B = np.eye(2 * l + 1)
    else:
        raise ValueError('Invalid from_order: ' + str(from_ordering))

    # Make sure we're using CS-phase (this should work for both real and complex bases)
    if from_cs == 'nocs':
        m = np.arange(-l, l + 1)
        B = ((-1.) ** (m * (m > 0)))[:, None] * B
    elif from_cs != 'cs':
        raise ValueError('Invalid from_cs: ' + str(from_cs))

    # If needed, change complex to real or real to complex
    # (we know how to do that in the centered, CS-phase bases)
    if from_field != to_field:
        if from_field == 'complex' and to_field == 'real':
            B = _cc2rc(l).dot(B)
        elif from_field == 'real' and to_field == 'complex':
            B = _cc2rc(l).conj().T.dot(B)
        else:
            raise ValueError('Invalid field:' + str(from_field) + ', ' + str(to_field))

    # If needed, change the normalization:
    if from_normalization != to_normalization:
        # First, change normalization to quantum
        if from_normalization == 'seismology':
            B = _seismology2quantum(l, full_matrix=False)[:, None] * B
        elif from_normalization == 'geodesy':
            B = _geodesy2quantum(l, full_matrix=False)[:, None] * B
        elif from_normalization == 'nfft':
            B = _nfft2quantum(l, full_matrix=False)[:, None] * B
        elif from_normalization != 'quantum':
            raise ValueError('Invalud from_normalization:' + str(from_normalization))

        # We're now in quantum normalization, change to output normalization
        if to_normalization == 'seismology':
            B = (1. / _seismology2quantum(l, full_matrix=False))[:, None] * B
        elif to_normalization == 'geodesy':
            B = (1. / _geodesy2quantum(l, full_matrix=False))[:, None] * B
        elif to_normalization == 'nfft':
            B = (1. / _nfft2quantum(l, full_matrix=False))[:, None] * B
        elif to_normalization != 'quantum':
            raise ValueError('Invalid to_normalization:' + str(to_normalization))

    #if from_field != to_field:
    #    if from_field == 'complex' and to_field == 'real':
    #        B = cc2rc(l).dot(B)
    #    elif from_field == 'real' and to_field == 'complex':
    #        #B = cc2rc(l).conj().T.dot(B)
    #        pass
    #    else:
    #        raise ValueError('Invalid field:' + str(from_field) + ', ' + str(to_field))
    #if to_field == 'real':
    #    B = cc2rc(l).dot(B)
    #elif to_field != 'complex':
    #    raise ValueError('Invalid to_field: ' + str(to_field))

    # Set the correct CS phase
    if to_cs == 'nocs':
        # We're in CS phase now, so cancel it:
        m = np.arange(-l, l + 1)
        B = ((-1.) ** (m * (m > 0)))[:, None] * B
    elif to_cs != 'cs':
        raise ValueError('Invalid to_cs: ' + str(to_cs))

    # If needed, change the order from centered:
    if to_ordering == 'block':
        B = _c2b(l).dot(B)
    elif to_ordering != 'centered':
        raise ValueError('Invalid to_ordering:' + str(to_ordering))

    return B


#TODO: make sure that change_of_basis_function accepts matrices, where each row is a vector to be changed of basis.
def change_of_basis_function(l, frm=('complex', 'seismology', 'centered', 'cs'),
                             to=('real', 'quantum', 'centered', 'cs')):
    """
    Return a function that will compute the change-of-basis that takes the 'frm' basis to the 'to' basis.
    Each basis is identified by:
     1) A field (real or complex)
     2) A normalization / phase convention ('seismology', 'quantum', or 'geodesy')
     3) An ordering convention ('centered', 'block')
     4) Whether to use Condon-Shortley phase (-1)^m for m > 0 ('cs', 'nocs')

    :param l: the weight (non-negative integer) of the irreducible representation, or an iterable of weights.
    :param frm: a 3-tuple (field, normalization, ordering) indicating the input basis.
    :param to: a 3-tuple (field, normalization, ordering) indicating the output basis.
    :return:
    """

    from_field, from_normalization, from_ordering, from_cs = frm
    to_field, to_normalization, to_ordering, to_cs = to

    if not isinstance(l, np.ndarray):  # collections.Iterable):
        l = np.atleast_1d(np.array(l))

    # First, bring us to the centered basis:
    if from_ordering == 'block':
        f1 = _b2c_func(l)
    elif from_ordering == 'centered':
        f1 = lambda x: x
    else:
        raise ValueError('Invalid from_order: ' + str(from_ordering))

    ms = np.zeros(np.sum(2 * l + 1), dtype=INT_TYPE)
    ls = np.zeros(np.sum(2 * l + 1), dtype=INT_TYPE)
    i = 0
    for ll in l:
        for mm in range(-ll, ll + 1):
            ms[i] = mm
            ls[i] = ll
            i += 1

    # Make sure we're using CS-phase (this should work for both real and complex bases)
    if from_cs == 'nocs':
        p = ((-1.) ** (ms * (ms > 0)))
        f2 = lambda x: f1(x) * p
    elif from_cs == 'cs':
        f2 = f1
    else:  # elif from_cs != 'cs':
        raise ValueError('Invalid from_cs: ' + str(from_cs))

    # If needed, change complex to real or real to complex
    # (we know how to do that in the centered, CS-phase bases)
    if from_field != to_field:
        if from_field == 'complex' and to_field == 'real':
            #B = _cc2rc(l).dot(B)
            #pos_m = m > 0
            #neg_m = m < 0
            #zero_m = m == 0
            #f3 = lambda x: r(f2(x) * _cc2rc_func(x, m), 3)
            f3 = lambda x: _cc2rc_func(f2(x), ms)
        elif from_field == 'real' and to_field == 'complex':
            f3 = lambda x: _rc2cc_func(f2(x), ms)
            #raise NotImplementedError('Real to complex not implemented yet')
        else:
            raise ValueError('Invalid field:' + str(from_field) + ', ' + str(to_field))
    else:
        f3 = f2

    # If needed, change the normalization:
    if from_normalization != to_normalization:
        # First, change normalization to quantum
        if from_normalization == 'seismology':
            f4 = lambda x: f3(x) * _seismology2quantum(l, full_matrix=False)
        elif from_normalization == 'geodesy':
            f4 = lambda x: f3(x) * _geodesy2quantum(l, full_matrix=False)
        elif from_normalization == 'nfft':
            f4 = lambda x: f3(x) * _nfft2quantum(l, full_matrix=False)
        elif from_normalization == 'quantum':
            f4 = f3
        else:  # elif from_normalization != 'quantum':
            raise ValueError('Invalud from_normalization:' + str(from_normalization))

        # We're now in quantum normalization, change to output normalization
        if to_normalization == 'seismology':
            f5 = lambda x: f4(x) / _seismology2quantum(l, full_matrix=False)
        elif to_normalization == 'geodesy':
            f5 = lambda x: f4(x) / _geodesy2quantum(l, full_matrix=False)
        elif to_normalization == 'nfft':
            f5 = lambda x: f4(x) / _nfft2quantum(l, full_matrix=False)
        elif to_normalization == 'quantum':
            f5 = f4
        else:  # elif to_normalization != 'quantum':
            raise ValueError('Invalid to_normalization:' + str(to_normalization))
    else:
        f5 = f3

    # Set the correct CS phase
    if to_cs == 'nocs':
        # We're in CS phase now, so cancel it:
        #m = np.arange(-l, l + 1)
        #B = ((-1.) ** (m * (m > 0)))[:, None] * B
        p = ((-1.) ** (ms * (ms > 0)))
        f6 = lambda x: f5(x) * p
    elif to_cs == 'cs':
        f6 = f5
    elif to_cs != 'cs':
        raise ValueError('Invalid to_cs: ' + str(to_cs))

    # If needed, change the order from centered:
    if to_ordering == 'block':
        #B = _c2b(l).dot(B)
        #raise NotImplementedError('Block basis not supported yet')
        f7 = lambda x: _c2b_func(l)(f6(x))
    elif to_ordering == 'centered':
        f7 = f6
    else:
        raise ValueError('Invalid to_ordering:' + str(to_ordering))

    return f7


def _cc2rc(l):
    """
    Compute change of basis matrix from the complex centered (cc) basis
    to the real centered (rc) basis.

    Let Y be a vector of complex spherical harmonics:
    Y = (Y^{-l}, ..., Y^0, ..., Y^l)^T
    Let S be a vector of real spherical harmonics as defined on the SH wiki page:
    S = (S^{-l}, ..., S^0, ..., S^l)^T
    Let B = cc2rc(l)
    Then S = B.dot(Y)

    B is a complex unitary matrix.

    Formula taken from:
    http://en.wikipedia.org/wiki/Spherical_harmonics#Real_form_2
    """

    B = np.zeros((2 * l + 1, 2 * l + 1), dtype=complex)
    for m in range(-l, l + 1):
        for n in range(-l, l + 1):
            row_ind = m + l
            col_ind = n + l
            if m == 0 and n == 0:
                B[row_ind, col_ind] = np.sqrt(2)
            if m > 0 and m == n:
                B[row_ind, col_ind] = (-1.) ** m
            elif m > 0 and m == -n:
                B[row_ind, col_ind] = 1.
            elif m < 0 and m == n:
                B[row_ind, col_ind] = 1j
            elif m < 0 and m == -n:
                B[row_ind, col_ind] = -1j * ((-1.) ** m)

    return (1.0 / np.sqrt(2)) * B


def _cc2rc_func(np.ndarray[COMPLEX_TYPE_t, ndim=1] x,
                np.ndarray[INT_TYPE_t, ndim=1] m_arr):
    """
    Compute change of basis from the complex centered (cc) basis
    to the real centered (rc) basis.

    Let Y be a vector of complex spherical harmonics:
    Y = (Y^{-l}, ..., Y^0, ..., Y^l)^T
    Let S be a vector of real spherical harmonics as defined on the SH wiki page:
    S = (S^{-l}, ..., S^0, ..., S^l)^T
    Let B = cc2rc(l)
    Then S = B.dot(Y)

    B is a complex unitary matrix.

    Formula taken from:
    http://en.wikipedia.org/wiki/Spherical_harmonics#Real_form_2
    """

    cdef int i = 0
    cdef np.ndarray[FLOAT_TYPE_t, ndim=1] x_out = np.empty(x.size)
    cdef double sq2 = np.sqrt(2)
    cdef double isq2 = 1. / sq2

    for i in range(m_arr.size):
        m = m_arr[i]
        if m > 0:
            x_out[i] = ((-1.) ** m * x[i] + x[i - 2 * m]).real * isq2
        elif m < 0:
            x_out[i] = (1j * x[i] - 1j * ((-1.) ** m) * x[i - 2 * m]).real * isq2
        else:
            x_out[i] = x[i].real

    return x_out


def _rc2cc_func(np.ndarray[FLOAT_TYPE_t, ndim=1] x,
                np.ndarray[INT_TYPE_t, ndim=1] m_arr):
    """
    Compute change of basis from the real centered (rc) basis
    to the complex centered (cc) basis.

    Formula taken from:
    http://en.wikipedia.org/wiki/Spherical_harmonics#Real_form_2
    """

    cdef int i = 0
    cdef np.ndarray[COMPLEX_TYPE_t, ndim=1] x_out = np.empty(x.size, dtype=COMPLEX_TYPE)
    cdef double sq2 = np.sqrt(2)
    cdef double isq2 = 1. / sq2

    for i in range(m_arr.size):
        m = m_arr[i]
        if m > 0:
            x_out[i] = ((-1.) ** m * x[i - 2 * m] * 1j + (-1.) ** m * x[i]) * isq2
        elif m < 0:
            x_out[i] = (-1j * x[i] + x[i - 2 * m]) * isq2
        else:
            x_out[i] = x[i]

    return x_out


def _c2b(l, full_matrix=True):
    """
    Compute change of basis matrix from the centered basis to
    the Pinchon-Hoggan block basis, in which the Pinchon-Hoggan J matrices
    are brought in block form.

    Let B = c2b(l)
    then B.dot(J_l).dot(B.T) is in block form with 4 blocks,
    as described by PH.
    """
    k = int(l) // 2
    if l % 2 == 0:
        # Permutation as defined by Pinchon-Hoggan for 1-based indices,
        # and l = 2 k
        sigma = np.array([2 * i for i in range(1, 2 * k + 1)]
                         + [2 * i - 1 for i in range(1, 2 * k + 2)])
    else:
        # Permutation as defined by Pinchon-Hoggan for 1-based indices,
        # and l = 2 k + 1
        sigma = np.array([2 * i for i in range(1, 2 * k + 2)]
                         + [2 * i - 1 for i in range(1, 2 * k + 3)])

    if full_matrix:
        # From permutation array sigma, create a permutation matrix B:
        B = np.zeros((2 * l + 1, 2 * l + 1))
        B[np.arange(2 * l + 1), sigma - 1] = 1.
        return B
    else:
        return sigma


def _c2b_func(l):
    """

    :param l:
    :return:
    """
    sigma = np.hstack([_c2b(li, full_matrix=False) - 1 for li in l])
    i_begin = 0
    for li in l:
        sigma[i_begin:i_begin + 2 * li + 1] += i_begin
        i_begin += 2 * li + 1
    f = lambda x: x[sigma]
    return f


def _b2c_func(l):
    sigma = np.hstack([_c2b(li, full_matrix=False) - 1 for li in l])
    i_begin = 0
    for li in l:
        sigma[i_begin:i_begin + 2 * li + 1] += i_begin
        i_begin += 2 * li + 1
    sigma_inv = np.argsort(sigma)
    f = lambda x: x[sigma_inv]
    return f




def _seismology2quantum(l, full_matrix=False):
    """

    :param l:
    :param full_matrix:
    :return:
    """
    if isinstance(l, collections.Iterable):
        diags = [_seismology2quantum(li, full_matrix=False) for li in l]
        diagonal = np.hstack(diags)

        if full_matrix:
            return np.diag(diagonal)
        else:
            return diagonal

    diagonal = (-np.ones(2 * l + 1)) ** np.arange(-l, l + 1)
    if full_matrix:
        return np.diag(diagonal)
    else:
        return diagonal


def _geodesy2quantum(l, full_matrix=False):
    if isinstance(l, collections.Iterable):
        diags = [_geodesy2quantum(li, full_matrix=False) for li in l]
        diagonal = np.hstack(diags)

        if full_matrix:
            return np.diag(diagonal)
        else:
            return diagonal

    diagonal = (-np.ones(2 * l + 1)) ** np.arange(-l, l + 1)
    diagonal /= np.sqrt(4 * np.pi)
    if full_matrix:
        return np.diag(diagonal)
    else:
        return diagonal


def _nfft2quantum(l, full_matrix=False):

    if isinstance(l, collections.Iterable):
        diags = [_nfft2quantum(li, full_matrix=False) for li in l]
        diagonal = np.hstack(diags)

        if full_matrix:
            return np.diag(diagonal)
        else:
            return diagonal

    diagonal = np.ones(2 * l + 1) * np.sqrt((2 * l + 1) / (4. * np.pi))
    # nfft only has (-1)^m phase factor for positive m; quantum has both pos and neg, so add phase factor to neg inds:
    # -> this is now done using a CS setting
    #m = np.arange(-l, l + 1)
    #diagonal *= ((-1) ** (m * (m < 0)))

    m = np.arange(-l, l + 1)
    diagonal *= (-1.) ** m

    if full_matrix:
        return np.diag(diagonal)
    else:
        return diagonal

"""
Code for rotating spherical harmonics expansions by the method described in:
Rotation matrices for real spherical harmonics: general rotations of atomic orbitals in space-fixed axes
D. Pinchon, P. E. Hoggan

All functions in this file assume that we're working in the basis of real, quantum-normalized, Pinchon-Hoggan block
spherical harmonics

This is NOT a user-facing API, so the interface of this file may change.
"""

import numpy as np
cimport numpy as np
cimport cython

from lie_learn.broadcasting import generalized_broadcast

# Load the J-matrices, which are stored in the same folder as this file
#import os
#Jb = np.load(os.path.join(os.path.dirname(__file__), 'J_block_0-478.npy'), allow_pickle=True)

FLOAT_TYPE = np.float64
ctypedef np.float64_t FLOAT_TYPE_t
INT_TYPE = np.int
ctypedef np.int_t INT_TYPE_t


def apply_rotation_block(g, X, irreps, c2b, J_block, l_max, X_out=None):

    X, g = generalized_broadcast([X, g])
    out_shape = X.shape
    X = X.reshape(-1, X.shape[-1]).copy()
    g = g.reshape(-1, g.shape[-1]).copy()

    if X_out is None:
        X_out = np.empty_like(X)
    X_out = X_out.reshape(-1, X.shape[-1])
    X_temp = np.empty_like(X_out)

    apply_z_rotation_block(g[:, 2], X, irreps, c2b, l_max, X_out=X_out)
    apply_J_block(X_out, J_block, X_out=X_temp)
    apply_z_rotation_block(g[:, 1], X_temp, irreps, c2b, l_max, X_out=X_out)
    apply_J_block(X_out, J_block, X_out=X_temp)
    apply_z_rotation_block(g[:, 0], X_temp, irreps, c2b, l_max, X_out=X_out)

    return X_out.reshape(out_shape)


@cython.wraparound(False)
@cython.nonecheck(False)
@cython.boundscheck(False)
cdef apply_z_rotation_block(np.ndarray[FLOAT_TYPE_t, ndim=1] angles,
                            np.ndarray[FLOAT_TYPE_t, ndim=2] X,
                            np.ndarray[INT_TYPE_t, ndim=1] irreps,
                            np.ndarray[INT_TYPE_t, ndim=1] c2b,
                            int l_max,
                            np.ndarray[FLOAT_TYPE_t, ndim=2] X_out):
    """
    Apply the rotation about the z-axis by angle angles[i] to the vector
    X[i, :] for all i. The vectors in X are assumed to be in the basis of real
    block spherical harmonics, corresponding to the irreps.

    In the *centered* basis, the z-axis rotation matrix (called X(angle) in the P&H paper) has a special
    form with cosines of different frequencies on the diagonal and sines on the
    anti-diagonal. This matrix is very sparse, so constructing it explicitly is
    very inefficient. This function applies a batch of such 'cross' matrices,
    represented implicitly by the corresponding angles, to a batch of vectors,
    without explicitly constructing the matrices.
    To do this in the block basis, we use a permutation array c2b that takes an index in the centered basis
    and returns the index for the block basis (which is a permutation of the centered basis).

    Args:
      angles: matrix of angles, shape (num_data, num_angles)
      irreps: a list of irrep weights (integers)
      c2b: an array of length dim, where c2b[i] is the index of centere basis vector i in the block basis
      l_max: an integer that is equal to np.max(irreps)
      X: matrix to be rotated, shape (num_data, dim)
      X_out: matrix where output will be stored, shape (dim, num_data, num_angles)

    Returns:
      X_out: matrix of shape (dim, num_data, num_angles), such that the
             vector X_out[:, i, j] is the rotation of X[:,i] by angles[i, j].
    """

    cdef int irrep_ind    # Index into the IRREPS2 matrix
    cdef int irrep        # The irrep weight
    cdef int center       # The index of the center element (l=0) of current irrep
    cdef int offset       # The offset of the current coordinate from the center
    cdef int abs_offset   # The absolute value of the offset
    cdef int offset_sign  # The sign of the offset
    #cdef int l_max = np.max(irreps)

    cdef int Xs0 = X.shape[0]
    cdef int Xs1 = X.shape[1]
    #cdef int Cs2

    cdef int vec
    cdef int coord
    cdef int angle_ind

    cdef int coord1_block
    cdef int coord2_block

    cdef np.ndarray[FLOAT_TYPE_t, ndim=2] C = np.cos(angles[None, :] * np.arange(l_max + 1)[:, None])
    cdef np.ndarray[FLOAT_TYPE_t, ndim=2] S = np.sin(angles[None, :] * np.arange(l_max + 1)[:, None])
    #Cs2 = C.shape[2]

    # Check that the irrep dimensions sum to the right dimensionality
    #assert (2 * irreps + 1).sum() == X.shape[0], "Invalid irrep dimensions"
    #assert angles.shape[0] == X.shape[1]
    #assert X_out.shape[0] == X.shape[0], "Invalid shape for X_out"
    #assert X_out.shape[1] == X.shape[1], "Invalid shape for X_out"
    #assert X_out.shape[2] == angles.shape[1], "Invalid shape for X_out"

    for vec in range(Xs0):  # Xs1):
        irrep_ind = 0   # Start at the first irrep
        irrep = irreps[irrep_ind]
        center = irrep
        for coord in range(Xs1):  # Xs0):  # X.shape[0]):
            offset = coord - center
            if offset > irrep:            # Finished with the current irrep?
                irrep_ind += 1             # Go to the next irrep
                irrep = irreps[irrep_ind]  # Get its weight from the list
                center = coord + irrep     # Compute the new center
                offset = -irrep            # equivalent to offset=coord-center;

            # Compute the absolute value and sign of the offset
            abs_offset = abs(offset)
            if offset >= 0:
                offset_sign = 1
            else:
                offset_sign = -1

            coord1_block = c2b[coord]
            coord2_block = c2b[center - offset]

            # Compute the value of the transformed coordinate
            # Note: we're always adding *two* values, even when offset=0 and hence there is only
            # one non-zero element in that row. This is not a problem because S2(0, vec) is always 0.
            #X_out[coord1_block, vec] = C[abs_offset, vec] * X[coord1_block, vec] \
            #                           - offset_sign * S[abs_offset, vec] * X[coord2_block, vec]
            X_out[vec, coord1_block] = C[abs_offset, vec] * X[vec, coord1_block] \
                                       - offset_sign * S[abs_offset, vec] * X[vec, coord2_block]


    return X_out


@cython.wraparound(False)
@cython.nonecheck(False)
@cython.boundscheck(False)
cdef apply_J_block(np.ndarray[FLOAT_TYPE_t, ndim=2] X,
                   list J_block,
                   np.ndarray[FLOAT_TYPE_t, ndim=2] X_out):
    """
    Multiply the Pinchon-Hoggan J matrix by a matrix X.

    This function uses the J matrix in the Pinchon-Hoggan block-basis. In this basis, the J-matrix is a block matrix
    with 4 blocks. This function performs this block-multiplication efficiently.

    :param X: numpy array of shape (dim, N) where dim is the total dimension of the representation and N is the number
     of vectors to be multiplied by J.
    :param J_block: list of list of precomputed J matrices in block form.
    :return:
    """

    cdef int l
    cdef int k
    cdef int rep_begin = 0
    cdef int b1s
    cdef int b1e
    cdef int b2s
    cdef int b2e
    cdef int b3s
    cdef int b3e
    cdef int b4s
    cdef int b4e
    cdef int li

    # Loop over irreps
    for li in range(len(J_block)):

        Jl = J_block[li]
        k = Jl[0].shape[0]
        l = (Jl[0].shape[0] + 2 * Jl[1].shape[0] + Jl[2].shape[0]) // 2

        # Determine block begin and end indices
        if l % 2 == 0:
            # blocks have dimension k, k, k, k+1
            b1s = rep_begin
            b1e = rep_begin + k
            b2s = rep_begin + k
            b2e = rep_begin + 2 * k
            b3s = rep_begin + 2 * k
            b3e = rep_begin + 3 * k
            b4s = rep_begin + 3 * k
            b4e = rep_begin + 4 * k + 1

            #b1 = Jbl[0:k, 0:k]
            #b2 = Jbl[k:2 * k, 2 * k:3 * k]
            ##b3 = Jbl[2 * k: 3 * k, k:2 * k]
            #b4 = Jbl[3 * k:, 3 * k:]
        else:
            # blocks have dimension k, k+1, k+1, k+1
            b1s = rep_begin
            b1e = rep_begin + k
            b2s = rep_begin + k
            b2e = rep_begin + 2 * k + 1
            b3s = rep_begin + 2 * k + 1
            b3e = rep_begin + 3 * k + 2
            b4s = rep_begin + 3 * k + 2
            b4e = rep_begin + 4 * k + 3

            #b1 = Jbl[0:k, 0:k]
            #b2 = Jbl[k:2 * k + 1, 2 * k + 1:3 * k + 2]
            ##b3 = Jbl[2 * k + 1:3 * k + 2, k:2 * k + 1]
            #b4 = Jbl[3 * k + 2:, 3 * k + 2:]

        # Multiply each block:
        X_out[:, b1s:b1e] = np.dot(Jl[0], X[:, b1s:b1e].T).T
        X_out[:, b2s:b2e] = np.dot(Jl[1], X[:, b3s:b3e].T).T
        X_out[:, b3s:b3e] = np.dot(Jl[1].T, X[:, b2s:b2e].T).T
        X_out[:, b4s:b4e] = np.dot(Jl[2], X[:, b4s:b4e].T).T

        rep_begin += 2 * l + 1

    return X_out


def make_c2b(irreps):
    # Centered to block basis

    # Maybe put this in separate file irrep_bases?
    c2b = np.empty((2 * irreps + 1).sum(), dtype=INT_TYPE)
    irrep_begin = 0
    for l in irreps:

        k = int(l) / 2
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

        sigma_inv = np.arange(0, 2 * l + 1)[np.argsort(sigma)]

        c2b[irrep_begin:irrep_begin + 2 * l + 1] = sigma_inv + irrep_begin
        irrep_begin += 2 * l + 1
    return c2b

from lie_learn.representations.SO3.spherical_harmonics import rsh

import numpy as np
cimport numpy as np


def estimate_spherical_quadrature_weights(sampling_set, max_bandwidth,
                                 normalization='quantum', condon_shortley=True,
                                 verbose=True):
    """

    :param sampling_set:
    :param max_bandwith:
    :return:
    """
    cdef int l
    cdef int m
    cdef int ll
    cdef int mm
    cdef int i

    cdef int M = sampling_set.shape[0]
    cdef int N = max_bandwidth
    cdef int N_total = (N + 1) ** 2  # = sum_l=0^N (2l + 1)

    cdef np.ndarray[np.float64_t, ndim=2] l_array = np.empty((N_total, 1))
    cdef np.ndarray[np.float64_t, ndim=2] m_array = np.empty((N_total, 1))

    theta = sampling_set[:, 0]
    phi = sampling_set[:, 1]

    if verbose:
        print 'Computing index arrays...'

    i = 0
    for l in range(N + 1):
        for m in range(-l, l + 1):
            l_array[i, 0] = l
            m_array[i, 0] = m
            i += 1

    if verbose:
        print 'Computing spherical harmonics...'
    Y = rsh(l_array, m_array, theta[None, :], phi[None, :],
            normalization=normalization, condon_shortley=condon_shortley)

    if verbose:
        print 'Computing least squares input'
    B = np.empty((N_total ** 2, M))
    t = np.empty(N_total ** 2)
    i = 0

    #print M, N, N_total
    #print theta[None, :].shape
    #print phi[None, :].shape
    #print Y.shape
    #print B.shape
    #print t.shape

    for l in range(N + 1):
        for m in range(-l, l + 1):
            rlm = Y[l ** 2 + l + m, :]
            for ll in range(N + 1):
                for mm in range(-ll, ll + 1):
                    B[i, :] = rlm * Y[ll ** 2 + ll + mm, :]
                    t[i] = float(ll == l and mm == m)
                    i += 1

    if verbose:
        print 'Computing least squares solution'
    return np.linalg.lstsq(B, t)

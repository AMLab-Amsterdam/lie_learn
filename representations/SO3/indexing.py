import numpy as np


def flat_ind_so3(l, m, n):
    """
    The SO3 spectrum consists of matrices f_hat^l of size (2l+1, 2l+1) for l=0, ..., L_max.
    If we flatten these matrices and stack them, we get a big vector where the element f_hat^l_mn has a certain 
     flat index. This function computes that index.

    The number of elements up to and including order L is
    N_L = sum_{l=0}^L (2l+1)^2 = 1/3 (2 L + 1) (2 L + 3) (L + 1)

    Element (l, m, n) has N_l elements before it from previous blocks, and in addition several elements in the current
    block. The number of elements in the current block, before (m, n) is determined as follows. 
    First we associate with indices m and n (running from -l to l) with their zero-based index:
    m' = m + l
    n' = n + l
    The linear index of this pair (m', n') is
    i = m' * w + n'
    where w is the width of the matrix, i.e. w = 2l + 1

    The final index of (l, m, n) is N_L + i

    :param l, m, n: spectral indices
    :return: index of (l, m, n) in flat vector
    """
    assert np.abs(m) <= l
    assert np.abs(n) <= l

    if l == 0:
        return 0  # The N_L formula only works for l > 0, so we special case this

    L = l - 1
    N_L = ((2 * L + 1) * (2 * L + 3) * (L + 1)) // 3
    i = (m + l) * (2 * l + 1) + (n + l)
    return N_L + i


def flat_ind_zp_so3(l, m, n, b):
    """
    The SO3 spectrum consists of matrices f_hat^l of size (2l+1, 2l+1) for l=0, ..., L_max = b - 1.
    These can be stored in a zero-padded array A of shape (b, 2b, 2b) with axes l, m, n with zero padding around
     the center of the last two axes. If we flatten this array A we get a vector v of size 4b^3.
    This function gives the flat index in this array v corresponding to element (l, m, n) 

    The zero-based 3D index of (l, m, n) in A is (l, b + m, b + n).
    The corresponding flat index is i = l * 4b^2 + (b + m) * 2b + b + n

    :param l, m, n: spectral indices
    :return: index of (l, m, n) in flat zero-padded vector 
    """
    return l * 4 * (b ** 2) + (b + m) * 2 * b + b + n


def list_to_flat(f_hat_list):
    """
    A function on the SO(3) spectrum can be represented as:
     1. a list f_hat of matrices f_hat[l] of size (2l+1, 2l+1)
     2. a flat vector which is the concatenation of the flattened matrices
     3. a zero-padded tensor with axes l, m, n.
     
    This function converts 1 to 2.
    
    :param f_hat: a list of matrices
    :return: a flat vector
    """
    return np.hstack([a.flat for a in f_hat_list])


def num_spectral_coeffs_up_to_order(b):
    """
    The SO(3) spectrum consists of matrices of size (2l+1, 2l+1) for l=0, ..., b - 1.
    This function computes the number of elements in a spectrum up to (but excluding) b - 1.
    
    The number of elements up to and including order L is
    N_L = sum_{l=0}^L (2l+1)^2 = 1/3 (2 L + 1) (2 L + 3) (L + 1)
    
    :param b: bandwidth 
    :return: the number of spectral coefficients
    """
    L_max = b - 1
    assert L_max >= 0
    return ((2 * L_max + 1) * (2 * L_max + 3) * (L_max + 1)) // 3


def flat_to_list(f_hat_flat, b):
    """
    A function on the SO(3) spectrum can be represented as:
     1. a list f_hat of matrices f_hat[l] of size (2l+1, 2l+1)
     2. a flat vector which is the concatenation of the flattened matrices
     3. a zero-padded tensor with axes l, m, n.
     
    This function converts 2 to 1.
    
    :param f_hat: a flat vector
    :return: a list of matrices
    """
    f_hat_list = []
    start = 0
    for l in range(b):
        f_hat_list.append(f_hat_flat[start:start + (2 * l + 1) ** 2].reshape(2 * l + 1, 2 * l + 1))
        start += (2 * l + 1) ** 2
    return f_hat_list

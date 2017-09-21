
import os
import numpy as np
import scipy.sparse as sp
from ..irrep_bases import change_of_basis_function, change_of_basis_matrix


def convert_all(num_mat_J_folder='/Users/user/Projects/LieLearn/SO3/shrot/legacy/pinchon2/Maple/NumMatJ/',
                out_folder='./', l_min=0, l_max=None):
    """
    Convert all numMatJ-XXX.dat files in a given folder to produce the following python pickle files:

    1) J_dense_0-[l_max].npy:
    2) J_sparse_0-[l_max].npy:
    3) J_block_0-[l_max].npy

    The numMatJ-XXX.dat files are the output of mkNumMatJ.mw

    :param num_mat_J_folder:
    :return:
    """
    J_dense = []
    J_sparse = []
    J_block = []

    if l_max is None:
        files = [f for f in os.listdir(num_mat_J_folder) if '.dat' in f]
        nums = [int(f[len("numMatJ-"):-len(".dat")]) for f in files]
        l_max = np.max(nums)
        print('Maximum l found:', l_max)

    for l in range(l_min, l_max + 1):
        print('Parsing l', l)

        filename = os.path.join(num_mat_J_folder, 'numMatJ-' + str(l) + '.dat')

        # Obtain the J matrix as a dense numpy array
        Jd = parse_J_single_l(filename)
        J_dense.append(Jd)
        J_sparse.append(sp.csr_matrix(Jd))

    print('Saving dense matrices...')
    np.save(os.path.join(out_folder, 'J_dense_' + str(l_min) + '-' + str(l_max)), J_dense)

    print('Saving sparse matrices...')
    np.save(os.path.join(out_folder, 'J_sparse_' + str(l_min) + '-' + str(l_max)), J_sparse)
    del J_sparse

    print('Converting to block basis...')
    J_block = make_block_J(J_dense)

    print('Saving block matrices...')
    np.save(os.path.join(out_folder, 'J_block_' + str(l_min) + '-' + str(l_max)), J_block)


def parse_J(file):
    """
    Loads the numMatJ.dat files provided by Pinchon & Hoggan into numpy matrices.
    I saved the result as a compressed pickle, so this function will
    only be needed when J matrices for larger l are needed.
    """

    f = open(file)
    lmax = int(f.readline().split(' ')[1])

    matj = [np.zeros((2 * l + 1, 2 * l + 1)) for l in range(lmax)]

    # Fill in matrix l=0 and l=1
    matj[0][0, 0] = 1.0
    matj[1][0, 1] = -1.0
    matj[1][1, 0] = -1.0
    matj[1][2, 2] = 1.0

    # Read out matrices >= 2
    for l in range(2, lmax):

        # Read and discard l-value:
        lval = int(f.readline().split(' ')[1])
        assert lval == l

        k = l/2
        k1 = k
        for i in range(k):
            for j in range(i):
                x = float(f.readline())
                matj[l][2*i+1, 2*j+1] = x
                matj[l][2*j+1, 2*i+1] = x
            x = float(f.readline())
            matj[l][2*i+1,2*i+1] = x

        if l != 2*k1:
            k += 1
        for i in range(k):
            for j in range(k):
                x = float(f.readline())
                matj[l][2*k1+2*j+1, 2*i] = x
                matj[l][2*i, 2*k1+2*j+1] = x

        if l == 2*k1:
            k += 1
        else:
            k1 += 1
        for i in range(k):
            for j in range(i):
                x = float(f.readline())
                matj[l][2*k1+2*i, 2*k1+2*j] = x
                matj[l][2*k1+2*j, 2*k1+2*i] = x
            x = float(f.readline())
            matj[l][2*k1+2*i, 2*k1+2*i] = x

    f.close()
    return matj


def parse_J_single_l(file):
    """
    Parse a single numMatJ-XXX.dat file produced by Pinchon & Hoggan's maple script mkNumMatJ.mw

    :param file:
    :return:
    """

    f = open(file)

    lval = int(f.readline().split(' ')[1])
    matj = np.zeros((2 * lval + 1, 2 * lval + 1))

    k = lval / 2
    k1 = k
    for i in range(k):
        for j in range(i):
            x = float(f.readline())
            matj[2*i+1, 2*j+1] = x
            matj[2*j+1, 2*i+1] = x
        x = float(f.readline())
        matj[2*i+1,2*i+1] = x

    if lval != 2 * k1:
        k += 1
    for i in range(k):
        for j in range(k):
            x = float(f.readline())
            matj[2*k1+2*j+1, 2*i] = x
            matj[2*i, 2*k1+2*j+1] = x

    if lval == 2 * k1:
        k += 1
    else:
        k1 += 1
    for i in range(k):
        for j in range(i):
            x = float(f.readline())
            matj[2*k1+2*i, 2*k1+2*j] = x
            matj[2*k1+2*j, 2*k1+2*i] = x
        x = float(f.readline())
        matj[2*k1+2*i, 2*k1+2*i] = x

    f.close()
    return matj


def make_block_J(Jd):
    """
    Convert a list of J matrices 0 to N, to block form.
    We change the basis on each J matrix (which is assumed to be in the real, quantum-normalized, centered basis,
    so that it is in the real, quantum-normalized, block basis.
    Then, we extract the blocks. There are 4 blocks for each irrep l, but the middle two are transposes of each other,
    so we store only 3 blocks. The outer two blocks are symmetric, but this is not exploited.

    :param Jd:
    :return:
    """
    Jb = []
    for l in range(len(Jd)):
        print('Converting to block matrix. (', l, 'of', len(Jd), ')')
        #Bl = c2b(l)
        #Jbl = Bl.dot(Jd[l]).dot(Bl.T)

        #c2b = change_of_basis_function(l,
        #                               frm=('real', 'quantum', 'centered', 'cs'),
        #                               to=('real', 'quantum', 'block', 'cs'))
        #Jbl = c2b(c2b(Jd[l]).T).T

        Bl = change_of_basis_matrix(l,
                                    frm=('real', 'quantum', 'centered', 'cs'),
                                    to=('real', 'quantum', 'block', 'cs'))
        Jbl = Bl.dot(Jd[l]).dot(Bl.T)

        k = l // 2
        if l % 2 == 0:
            # blocks have dimension k, k, k, k+1
            b1 = Jbl[0:k, 0:k]
            b2 = Jbl[k:2 * k, 2 * k:3 * k]
            #b3 = Jbl[2 * k: 3 * k, k:2 * k]
            b4 = Jbl[3 * k:, 3 * k:]
        else:
            # blocks have dimension k, k+1, k+1, k+1
            b1 = Jbl[0:k, 0:k]
            b2 = Jbl[k:2 * k + 1, 2 * k + 1:3 * k + 2]
            #b3 = Jbl[2 * k + 1:3 * k + 2, k:2 * k + 1]
            b4 = Jbl[3 * k + 2:, 3 * k + 2:]
        Jb.append([b1, b2, b4])
    return Jb

import numpy as np

import spaces.Rn as Rn

parameterizations = ('rotation-translation', '2x3 matrix', '3x3 matrix')


def compose(g, h, parameterization=None, g_parameterization=None, h_parameterization=None, out_parameterization=None):
    """
    Compose elements g, h in SE(2).
    """
    if parameterization is not None:
        g_parameterization = parameterization
        h_parameterization = parameterization
        out_parameterization = parameterization

    g_mat = change_parameterization(g, p_from=g_parameterization, p_to='3x3 matrix')
    h_mat = change_parameterization(h, p_from=h_parameterization, p_to='3x3 matrix')
    gh_mat = np.einsum('...ij,...jk->...ik', g_mat, h_mat)
    return change_parameterization(g=gh_mat, p_from='3x3 matrix', p_to=out_parameterization)

def invert(g, parameterization):
    """
    Invert element g in SE(2), where g can have any supported parameterization.
    """

    # Change to (theta, tau1, tau2) paramterization.
    g_rt = change_parameterization(g, p_from=parameterization, p_to='rotation-translation')
    g_inv_rt = np.empty_like(g_rt)
    g_inv_rt[..., 0] = -g_rt[..., 0]
    g_inv_rt[..., 1] = -(np.cos(-g_rt[..., 0]) * g_rt[..., 1] - np.sin(-g_rt[..., 0]) * g_rt[..., 2])
    g_inv_rt[..., 2] = -(np.sin(-g_rt[..., 0]) * g_rt[..., 1] + np.cos(-g_rt[..., 0]) * g_rt[..., 2])

    return change_parameterization(g=g_inv_rt, p_from='rotation-translation', p_to=parameterization)


def transform(g, g_parameterization, x, x_parameterization):
    """
    Apply rotation g in SE(2) to points x.
    """
    g_3x3 = change_parameterization(g, p_from=g_parameterization, p_to='3x3 matrix')
    x_homvec = Rn.change_coordinates(x, n=2, p_from=x_parameterization, p_to='homogeneous')
    #gx_homvec = g_3x3.dot(x_homvec)
    gx_homvec = np.einsum('...ij,...j->...i', g_3x3, x_homvec)
    return Rn.change_coordinates(gx_homvec, n=2, p_from='homogeneous', p_to=x_parameterization)


def change_parameterization(g, p_from, p_to):

    g = np.array(g)

    if p_from == p_to:
        return g

    if p_from == 'rotation-translation' and p_to == '2x3 matrix':
        g_out = np.empty(g.shape[:-1] + (2, 3))
        g_out[..., 0, 0] = np.cos(g[..., 0])
        g_out[..., 0, 1] = -np.sin(g[..., 0])
        g_out[..., 1, 0] = np.sin(g[..., 0])
        g_out[..., 1, 1] = np.cos(g[..., 0])
        g_out[..., 0, 2] = g[..., 1]
        g_out[..., 1, 2] = g[..., 2]
        return g_out

    if p_from == 'rotation-translation' and p_to == '3x3 matrix':
        g_out = np.empty(g.shape[:-1] + (3, 3))
        g_out[..., 0, 0] = np.cos(g[..., 0])
        g_out[..., 0, 1] = -np.sin(g[..., 0])
        g_out[..., 0, 2] = g[..., 1]
        g_out[..., 1, 0] = np.sin(g[..., 0])
        g_out[..., 1, 1] = np.cos(g[..., 0])
        g_out[..., 1, 2] = g[..., 2]
        g_out[..., 2, 0] = 0.
        g_out[..., 2, 1] = 0.
        g_out[..., 2, 2] = 1.
        return g_out

    else:
        raise ValueError('Not supported (yet):' + str(p_from) + ' to ' + str(p_to))
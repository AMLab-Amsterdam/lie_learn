
import numpy as np

parameterizations = ('MAT', 'C', 'ANG')

def compose(g, h, parameterization='MAT', g_parameterization=None, h_parameterization=None, out_parameterization=None):
    """
    Compose elements g, h in SO(2).
    g and h can have the following parameterizations:
    1: MAT   2x2 rotation matrix
    2: C     1x1  complex exponential (z=exp(i theta))
    """
    if parameterization is not None:
        g_parameterization = parameterization
        h_parameterization = parameterization
        out_parameterization = parameterization

    g_mat = change_parameterization(g, p_from=g_parameterization, p_to='MAT')
    h_mat = change_parameterization(h, p_from=h_parameterization, p_to='MAT')
    gh_mat = np.einsum('...ij,...jk->...ik', g_mat, h_mat)
    return change_parameterization(g=gh_mat, p_from='MAT', p_to=out_parameterization)

def invert(g, parameterization):
    """
    Invert element g in SO(2), where g can have any supported parameterization:
    1: MAT   2x2 rotation matrix
    2: C     1x1  complex exponential (z=exp(i theta))
    """

    g_mat = change_parameterization(g, p_from=parameterization, p_to='MAT')
    g_mat_T = g_mat.transpose(list(range(0, g_mat.ndim - 2)) + [g_mat.ndim - 1, g_mat.ndim - 2])  # Transpose last axes
    return change_parameterization(g=g_mat_T, p_from='MAT', p_to=parameterization)

def transform(g, g_parameterization, x, x_parameterization):
    """
    Apply rotation g in SO(2) to points x.
    """
    #g_mat = change_parameterization(g_parameterization + 'toMAT', g, ichk=0)
    #x_vec = change_coordinates(x_parameterization + 'toC', x)
    #gx_vec = g_mat.dot(x_vec)
    #return change_coordinates('Cto' + x_parameterization, gx_vec)
    raise NotImplementedError('SO2 transform not implemented')


def change_parameterization(g, p_from, p_to):
    """

    """
    if p_from == p_to:
        return g

    elif p_from == 'MAT' and p_to == 'C':
        theta = np.arctan2(g[..., 1, 0], g[..., 0, 0])
        return np.exp(1j * theta)
    elif p_from == 'MAT' and p_to == 'ANG':
        return  np.arctan2(g[..., 1, 0], g[..., 0, 0])
    elif p_from == 'C'  and p_to == 'MAT':
        theta = np.angle(g)
        c = np.cos(theta)
        s = np.sin(theta)
        return np.array([[c, -s], [s, c]]).transpose(list(range(2, 2 + c.ndim)) + [0, 1])
    elif p_from == 'C' and p_to == 'ANG':
        return np.angle(g)
    elif p_from == 'ANG' and p_to == 'MAT':
        c = np.cos(g)
        s = np.sin(g)
        return np.array([[c, -s], [s, c]]).transpose(list(range(2, 2 + c.ndim)) + [0, 1])
    elif p_from == 'ANG' and p_to == 'C':
        return np.exp(1j * g)
    else:
        raise ValueError('Unsupported conversion:' + str(p_from) + ' to ' + str(p_to))
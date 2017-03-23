"""
n-dimensional real space, R^n.
"""


import numpy as np

# The following functions are part of the public interface of this module;
# other spaces / groups define their own meshgrid and linspace functions that work in an analogous way;
# for R^n the standard numpy functions fulfill this role.
from numpy import meshgrid, linspace


def change_coordinates(coords, n, p_from='C', p_to='S'):
    """
    Change Spherical to Cartesian coordinates and vice versa.

    todo: make this work for R^n and not just R^2, R^3

    :param conversion:
    :param coords:
    :return:
    """

    coords = np.asarray(coords)

    if p_from == p_to:
        return coords

    if n == 2:
        if (p_from == 'P' or p_from == 'polar') and (p_to == 'C' or p_to == 'cartesian'):
            r = coords[..., 0]
            theta = coords[..., 1]
            out = np.empty_like(coords)
            out[..., 0] = r * np.cos(theta)
            out[..., 1] = r * np.sin(theta)
            return out
        elif (p_from == 'C' or p_from == 'cartesian') and (p_to == 'P' or p_to == 'polar'):
            x = coords[..., 0]
            y = coords[..., 1]
            out = np.empty_like(coords)
            out[..., 0] = np.sqrt(x ** 2 + y ** 2)
            out[..., 1] = np.arctan2(y, x)
            return out
        elif (p_from == 'C' or p_from == 'cartesian') and (p_to == 'H' or p_to == 'homogeneous'):
            x = coords[..., 0]
            y = coords[..., 1]
            out = np.empty(coords.shape[:-1] + (3,))
            out[..., 0] = x
            out[..., 1] = y
            out[..., 2] = 1.
            return out
        elif (p_from == 'H' or p_from == 'homogeneous') and (p_to == 'C' or p_to == 'cartesian'):
            xc = coords[..., 0]
            yc = coords[..., 1]
            c = coords[..., 2]
            out = np.empty(coords.shape[:-1] + (2,))
            out[..., 0] = xc / c
            out[..., 1] = yc / c
            return out
        else:
            raise ValueError('Unknown conversion' + str(p_from) + ' to ' + str(p_to))

    elif n == 3:

        if p_from == 'S' and p_to == 'C':

            theta = coords[..., 0]
            phi = coords[..., 1]
            r = coords[..., 2]

            out = np.empty(theta.shape + (3,))

            ct = np.cos(theta)
            cp = np.cos(phi)
            st = np.sin(theta)
            sp = np.sin(phi)
            out[..., 0] = r * st * cp  # x
            out[..., 1] = r * st * sp  # y
            out[..., 2] = r * ct       # z
            return out

        elif p_from == 'C' and p_to == 'S':

            x = coords[..., 0]
            y = coords[..., 1]
            z = coords[..., 2]

            out = np.empty_like(coords)
            out[..., 2] = np.sqrt(x ** 2 + y ** 2 + z ** 2)  # r
            out[..., 0] = np.arccos(z / out[..., 2])         # theta
            out[..., 1] = np.arctan2(y, x)                   # phi
            return out

        else:
            raise ValueError('Unknown conversion:' + str(p_from) + ' to ' + str(p_to))
    else:
        raise ValueError('Only dimension n=2 and n=3 supported for now.')


def linspace(b, convention):

    pass




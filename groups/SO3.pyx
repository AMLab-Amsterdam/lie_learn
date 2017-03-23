
# # Notes
# Rotation conventions consist of
# -> orientation of frame: (left- or right-handed) associate x,y,z with thumb, index, middle finger resp. of r or l hand
# -> active or passive / frame-fixed or body-fixed / alibi or alias

# This code uses a Right-hand rule
# (i.e. positive Euler angles indicate clockwise rotations when looking in the positive direction
# along an axis) in a right-handed coordinate frame (thumb, index, middle finger = x, y, z, respectively)
# Furthermore, this code uses a frame-rotation / vector-fixed / alias convention instead of
# the vector-rotation / frame-fixed / alibi convention.
# To obtain the other, simply invert the rotation operator.
##

# Todo:
# -Use quaternions in compose_rotation because they're more stable and cheaper to compose. (do check if this works..)
# -Make compose broadcast as a gufunc
# -change_parameterization could be further optimized
# -implement fix the input checking code in change_parameterizations
# -detect singularities and handle them correctly, so that the output is always acceptable

import numpy as np
cimport numpy as np
import lie_learn.spaces.rn as Rn

FLOAT_TYPE = np.float64
ctypedef np.float64_t FLOAT_TYPE_t

parameterizations = ('MAT', 'Q', 'EV',
                     'EA123', 'EA132', 'EA213', 'EA231', 'EA312', 'EA321',  # Type-I Euler angles AKA Tait-Bryan angles
                     'EA121', 'EA131', 'EA212', 'EA232', 'EA313', 'EA323')  # Type-II Euler angles

def compose(g, h, parameterization='MAT', g_parameterization=None, h_parameterization=None, out_parameterization=None):
    """
    Compose (i.e. multiply) elements g, h in SO(3).
    g and h can have any parameterization supported by our adaptation of the SpinCalc function:
    1: Q        Rotation Quaternions
    2: EV       Euler Vector and rotation angle (degrees)
    3: MAT      Orthogonal MAT Rotation Matrix
    4: EAXXX    Euler angles (12 possible sets) (degrees)
    """
    if parameterization is not None:
        g_parameterization = parameterization
        h_parameterization = parameterization
        out_parameterization = parameterization

    g_mat = change_coordinates(g=g, p_from=g_parameterization, p_to='MAT', perform_checks=False)
    h_mat = change_coordinates(g=h, p_from=h_parameterization, p_to='MAT', perform_checks=False)
    gh_mat = np.einsum('...ij,...jk->...ik', g_mat, h_mat)
    return change_coordinates(g=gh_mat, p_from='MAT', p_to=out_parameterization, perform_checks=False)

def invert(g, parameterization='MAT'):
    """
    Invert element g in SO(3), where g can have any supported parameterization:
    1: Q        Rotation Quaternions
    2: EV       Euler Vector and rotation angle (degrees)
    3: MAT      Orthogonal MAT Rotation Matrix
    4: EAXXX    Euler angles (12 possible sets) (degrees)
    """
    g_mat = change_coordinates(g=g, p_from=parameterization, p_to='MAT', perform_checks=False)
    g_mat_T = g_mat.transpose(list(range(0, g_mat.ndim - 2)) + [g_mat.ndim - 1, g_mat.ndim - 2])  # Transpose last axes
    return change_coordinates(g=g_mat_T, p_from='MAT', p_to=parameterization, perform_checks=False)

def transform_r3(g, x, g_parameterization='MAT', x_parameterization='C'):
    """
    Apply rotation g in SO(3) to points x in R^3.
    """
    g_mat = change_coordinates(g=g, p_from=g_parameterization, p_to='MAT', perform_checks=False)
    x_vec = Rn.change_coordinates(x, n=3, p_from=x_parameterization, p_to='C')
    #gx_vec = g_mat.dot(x_vec)
    gx_vec = np.einsum('...ij,...j->...i', g_mat, x_vec)
    return Rn.change_coordinates(gx_vec, n=3, p_from = 'C', p_to=x_parameterization)

def change_coordinates(g, p_from, p_to, units='rad', perform_checks=False):
    """
    Change the coordinates of rotation operators g_in in SO(3).

    Parameterizations:
    MAT - 3x3 matrix which pre-multiplies by a coordinate frame vector to rotate it to the desired new frame.
    EAXXX - [psi,theta,phi] (3,) vector list dictating to the first angle rotation (psi), the second (theta),
            and third (phi). XXX stands for the axes of rotation, in order (e.g. XXX=131)
    EV - [m1,m2,m3,MU] (4,) vector list dictating the components of euler rotation vector (original coordinate frame),
         and the Euler rotation angle about that vector (MU)
    Q - [q1,q2,q3,q4] (4,) row vector list defining quaternion of rotation.
        q4 = np.cos(MU/2) where MU is Euler rotation angle

    :param p_from: parameterization of the input g_in
    :param p_to: parameterization to convert to
    :param g: input transformations, shape s_input + s_p_from, where s_input is an arbitrary shape
           and s_p_from is the shape a single input rotation in the p_from parameterization.
           e.g. (3, 3) for rotation matrices, (4,) for quaternions and Euler vectors, and (3,) for Euler angles.
    :param units: 'rad' or 'deg'
    :param perform_checks: if True (default) this function will perform checks on the input
    :return: array shape (s_input, s_p_to)
    """
    g = np.asarray(g)

    if perform_checks:
        raise NotImplementedError('Need to update checks code to handle arrays g_in with many elements')
        if p_from[:2] == 'EA':  # if input type is Euler angles, determine the order of rotations
            if units == 'deg':
                # If input is in degrees convert it to radians.
                # It will be converted back to degrees for output.
                g = g * np.pi / 180.

            rot_1_in = int(p_from[2])
            rot_2_in = int(p_from[3])
            rot_3_in = int(p_from[4])

            # check that all orders are between 1 and 3
            if rot_1_in < 1 or rot_2_in < 1 or rot_3_in < 1 or rot_1_in > 3 or rot_2_in > 3 or rot_3_in > 3:
                raise ValueError('Invalid input Euler angle order type (conversion string)')
            # check that no 2 consecutive orders are equal (invalid)
            elif rot_1_in == rot_2_in or rot_2_in == rot_3_in:
                raise ValueError('Invalid input Euler angle order type (conversion string)')

            # check first input dimensions
            if g.shape[-1] != 3:
                raise ValueError('Input euler angle data vector should have last dimension of length 3')

            # identify np.singularities
            if rot_1_in == rot_3_in:  # Type 2 rotation (first and third rotations about same axis)
                if (g[1] <= 0) or (g[1] >= np.pi):  # confirm second angle within range
                    raise ValueError('Second input Euler angle(s) outside 0 to 180 degree range')
                elif (np.abs(g[1]) < 2) or np.abs((g[1]) > (np.pi - 0.035)):  # check for np.singularity 88 deg
                    if perform_checks:
                        print('Warning: Input Euler angle rotation(s) near a singularity. ' +
                              'Second angle near 0 or 180 degrees.')

            else:  # Type 1 rotation (all rotations about each of three axes)
                if np.abs(g[1]) >= np.pi / 2:  # confirm second angle within range
                    raise ValueError('Second input Euler angle(s) outside -90 to 90 degree range')
                elif np.abs(g[1]) > (np.pi / 2 - 0.035):  # check for np.singularity
                    if perform_checks:
                        print('Warning: Input Euler angle(s) rotation near a singularity. ' +
                              'Second angle near -90 or 90 degrees.')

        if p_from == 'MAT':
            if g.shape != (3, 3):
                raise ValueError('Matrix is not 3x3. It is:' + str(g.shape))

            # Check if matrix is indeed special orthogonal
            if not np.isclose(np.sum(np.abs(g.dot(g.T) - np.eye(3))), 0):
                raise ValueError('g_in not orthogonal')
            if not np.isclose(np.abs(np.linalg.det(g) - 1), 0.0):
                raise ValueError('g_in not a proper rotation')

        if p_from == 'EV':  # Euler Vector Input Type
            if g.shape[0] != 4:  # or INPUT.shape[2]!=1:   # check dimensions
                raise ValueError('Input euler vector and rotation data matrix is not Nx4')

            if units == 'deg':
                mu = g[3] * np.pi / 180  # assign mu name for clarity
            else:
                mu = g[3]

            # check that input m's constitute unit vector
            if not np.isclose(np.sum(g ** 2), 1.0):
                raise ValueError('Input Euler vector is not unit length')

            if mu < 0 or mu > 2 * np.pi:  # check if rotation about euler vector is between 0 and 360
                print('Warning: Input euler rotation angle(s) not between 0 and 360 degrees')

        if p_from == 'Q':
            if g.shape[0] != 4:
                raise ValueError('Input quaternion matrix is not 4xN')

            if not np.isclose(np.abs(np.sqrt(g[0] ** 2 + g[1] ** 2 + g[2] ** 2 + g[3] ** 2)), 1.):
                print('Warning: Input quaternion norm(s) deviate(s) from unity by more than tolerance')

        if p_to[:2] == 'EA':  # if output type is Euler angles, determine order of rotations
            rot_1_out = int(p_to[2])
            rot_2_out = int(p_to[3])
            rot_3_out = int(p_to[4])
            if rot_1_out < 1 or rot_2_out < 1 or rot_3_out < 1 or rot_1_out > 3 or rot_2_out > 3 or rot_3_out > 3:
                raise ValueError('Invalid output Euler angle order type (conversion string).')
            elif rot_1_out == rot_2_out or rot_2_out == rot_3_out:
                raise ValueError('Invalid output Euler angle order type (conversion string).')

    # Convert inputs to quaternions.
    # The output will be calculated in the second portion of the code from these quaternions.
    if p_from == 'MAT':
        in_shape = g.shape[:-2]
        g = g.reshape(-1, 3, 3)

        q = rotation_matrix_to_quaternion(g)

    elif p_from == 'EV':  # Euler Vector Input Type
        in_shape = g.shape[:-1]
        g = g.reshape(-1, 4)

        if units == 'deg':
            mu = g[:, 3] * np.pi / 180
        else:
            mu = g[:, 3]

        # Construct the quaternion:
        half_mu = 0.5 * mu
        s = np.sin(half_mu)
        c = np.cos(half_mu)
        q = np.asarray([g[:, 0] * s,
                        g[:, 1] * s,
                        g[:, 2] * s,
                        c]).T

    elif p_from[:2] == 'EA':
        in_shape = g.shape[:-1]
        g = g.reshape(-1, 3)

        if units == 'deg':
            # convert from deg to radians if necessary
            psi = g[:, 0] * np.pi / 180.
            theta = g[:, 1] * np.pi / 180.
            phi = g[:, 2] * np.pi / 180.
        else:
            psi = g[:, 0]
            theta = g[:, 1]
            phi = g[:, 2]

        # Pre-calculate cosines and sines of the half-angles for conversion.
        c1 = np.cos(0.5 * psi)
        c2 = np.cos(0.5 * theta)
        c3 = np.cos(0.5 * phi)
        s1 = np.sin(0.5 * psi)
        s2 = np.sin(0.5 * theta)
        s3 = np.sin(0.5 * phi)
        c13 = np.cos(0.5 * (psi + phi))
        s13 = np.sin(0.5 * (psi + phi))
        c1_3 = np.cos(0.5 * (psi - phi))
        s1_3 = np.sin(0.5 * (psi - phi))
        c3_1 = np.cos(0.5 * (phi - psi))
        s3_1 = np.sin(0.5 * (phi - psi))
        p_from_axis_order = int(p_from[2:])
        if p_from_axis_order == 121:
            q = np.asarray([c2 * s13, s2 * c1_3, s2 * s1_3, c2 * c13]).T
        elif p_from_axis_order == 232:
            q = np.asarray([s2 * s1_3, c2 * s13, s2 * c1_3, c2 * c13]).T
        elif p_from_axis_order == 313:
            q = np.asarray([s2 * c1_3, s2 * s1_3, c2 * s13, c2 * c13]).T
        elif p_from_axis_order == 131:
            q = np.asarray([c2 * s13, s2 * s3_1, s2 * c3_1, c2 * c13]).T
        elif p_from_axis_order == 212:
            q = np.asarray([s2 * c3_1, c2 * s13, s2 * s3_1, c2 * c13]).T
        elif p_from_axis_order == 323:
            q = np.asarray([s2 * s3_1, s2 * c3_1, c2 * s13, c2 * c13]).T
        elif p_from_axis_order == 123:
            q = np.asarray([s1 * c2 * c3 + c1 * s2 * s3, c1 * s2 * c3 - s1 * c2 * s3,
                            c1 * c2 * s3 + s1 * s2 * c3, c1 * c2 * c3 - s1 * s2 * s3]).T
        elif p_from_axis_order == 231:
            q = np.asarray([c1 * c2 * s3 + s1 * s2 * c3, s1 * c2 * c3 + c1 * s2 * s3,
                            c1 * s2 * c3 - s1 * c2 * s3, c1 * c2 * c3 - s1 * s2 * s3]).T
        elif p_from_axis_order == 312:
            q = np.asarray([c1 * s2 * c3 - s1 * c2 * s3, c1 * c2 * s3 + s1 * s2 * c3,
                            s1 * c2 * c3 + c1 * s2 * s3, c1 * c2 * c3 - s1 * s2 * s3]).T
        elif p_from_axis_order == 132:
            q = np.asarray([s1 * c2 * c3 - c1 * s2 * s3, c1 * c2 * s3 - s1 * s2 * c3,
                            c1 * s2 * c3 + s1 * c2 * s3, c1 * c2 * c3 + s1 * s2 * s3]).T
        elif p_from_axis_order == 213:
            q = np.asarray([c1 * s2 * c3 + s1 * c2 * s3, s1 * c2 * c3 - c1 * s2 * s3,
                            c1 * c2 * s3 - s1 * s2 * c3, c1 * c2 * c3 + s1 * s2 * s3]).T
        elif p_from_axis_order == 321:
            q = np.asarray([c1 * c2 * s3 - s1 * s2 * c3, c1 * s2 * c3 + s1 * c2 * s3,
                            s1 * c2 * c3 - c1 * s2 * s3, c1 * c2 * c3 + s1 * s2 * s3]).T
        else:
            raise ValueError('Invalid input Euler angle order type (conversion string)')

    elif p_from == 'Q':
        in_shape = g.shape[:-1]
        g = g.reshape(-1, 4)

        q = g

    # Normalize quaternions in case of deviation from unity.
    # User has already been warned of deviation.
    q = q / np.sqrt((q ** 2).sum(axis=1))[:, None]

    # Convert the quaternion that represents g_in to the desired output parameterization
    if p_to == 'MAT':
        output = np.asarray([q[:, 0] ** 2 - q[:, 1] ** 2 - q[:, 2] ** 2 + q[:, 3] ** 2,
                             2 * (q[:, 0] * q[:, 1] + q[:, 2] * q[:, 3]),
                             2 * (q[:, 0] * q[:, 2] - q[:, 1] * q[:, 3]),
                             2 * (q[:, 0] * q[:, 1] - q[:, 2] * q[:, 3]),
                             -q[:, 0] ** 2 + q[:, 1] ** 2 - q[:, 2] ** 2 + q[:, 3] ** 2,
                             2 * (q[:, 1] * q[:, 2] + q[:, 0] * q[:, 3]),
                             2 * (q[:, 0] * q[:, 2] + q[:, 1] * q[:, 3]),
                             2 * (q[:, 1] * q[:, 2] - q[:, 0] * q[:, 3]),
                             -q[:, 0] ** 2 - q[:, 1] ** 2 + q[:, 2] ** 2 + q[:, 3] ** 2])
        output = output.reshape((3, 3, -1)).transpose(2, 0, 1)
        output = output.reshape(in_shape + (3, 3))

    elif p_to == 'EV':
        mu = 2 * np.arctan2(np.sqrt((q[:, 0:3] ** 2).sum(axis=1)), q[:, 3])
        N = q.shape[0]
        output = np.empty((N, 4))
        for i in range(N):
            if np.sin(0.5 * mu[i]) != 0:
                if units == 'deg':
                    output[i, :] = np.asarray([q[i, 0] / np.sin(0.5 * mu[i]),
                                               q[i, 1] / np.sin(0.5 * mu[i]),
                                               q[i, 2] / np.sin(0.5 * mu[i]),
                                               mu[i] * 180. / np.pi])
                else:
                    output[i, :] = np.asarray([q[i, 0] / np.sin(0.5 * mu[i]),
                                               q[i, 1] / np.sin(0.5 * mu[i]),
                                               q[i, 2] / np.sin(0.5 * mu[i]),
                                               mu[i]])
            else:
                if units == 'deg':
                    output[i, :] = [1, 0, 0, mu[i] * 180. / np.pi]
                else:
                    output[i, :] = [1, 0, 0, mu[i]]
        output = output.reshape(in_shape + (4,))

    elif p_to == 'Q':
        output = q
        output = output.reshape(in_shape + (4,))

    elif p_to[:2] == 'EA':
        p_to_axis_order = int(p_to[2:])
        if p_to_axis_order == 121:
            psi = np.arctan2(q[:, 0] * q[:, 1] + q[:, 2] * q[:, 3], q[:, 1] * q[:, 3] - q[:, 0] * q[:, 2])
            theta = np.arccos(np.clip(q[:, 3] ** 2 + q[:, 0] ** 2 - q[:, 1] ** 2 - q[:, 2] ** 2, -1.0, 1.0))
            phi = np.arctan2(q[:, 0] * q[:, 1] - q[:, 2] * q[:, 3],
                             q[:, 0] * q[:, 2] + q[:, 1] * q[:, 3])
            euler_type = 2
        elif p_to_axis_order == 232:
            psi = np.arctan2(q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2], q[:, 2] * q[:, 3] - q[:, 0] * q[:, 1])
            theta = np.arccos(np.clip(q[:, 3] ** 2 - q[:, 0] ** 2 + q[:, 1] ** 2 - q[:, 2] ** 2, -1.0, 1.0))
            phi = np.arctan2(q[:, 1] * q[:, 2] - q[:, 0] * q[:, 3], q[:, 0] * q[:, 1] + q[:, 2] * q[:, 3])
            euler_type = 2
        elif p_to_axis_order == 313:
            psi = np.arctan2(q[:, 0] * q[:, 2] + q[:, 1] * q[:, 3], q[:, 0] * q[:, 3] - q[:, 1] * q[:, 2])
            theta = np.arccos(np.clip(q[:, 3] ** 2 - q[:, 0] ** 2 - q[:, 1] ** 2 + q[:, 2] ** 2, -1.0, 1.0))
            phi = np.arctan2(q[:, 0] * q[:, 2] - q[:, 1] * q[:, 3], q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2])
            euler_type = 2
        elif p_to_axis_order == 131:
            psi = np.arctan2(q[:, 0] * q[:, 2] - q[:, 1] * q[:, 3], q[:, 0] * q[:, 1] + q[:, 2] * q[:, 3])
            theta = np.arccos(np.clip(q[:, 3] ** 2 + q[:, 0] ** 2 - q[:, 1] ** 2 - q[:, 2] ** 2, -1.0, 1.0))
            phi = np.arctan2(q[:, 0] * q[:, 2] + q[:, 1] * q[:, 3], q[:, 2] * q[:, 3] - q[:, 0] * q[:, 1])
            euler_type = 2
        elif p_to_axis_order == 212:
            psi = np.arctan2(q[:, 0] * q[:, 1] - q[:, 2] * q[:, 3], q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2])
            theta = np.arccos(np.clip(q[:, 3] ** 2 - q[:, 0] ** 2 + q[:, 1] ** 2 - q[:, 2] ** 2, -1.0, 1.0))
            phi = np.arctan2(q[:, 0] * q[:, 1] + q[:, 2] * q[:, 3], q[:, 0] * q[:, 3] - q[:, 1] * q[:, 2])
            euler_type = 2
        elif p_to_axis_order == 323:
            psi = np.arctan2(q[:, 1] * q[:, 2] - q[:, 0] * q[:, 3], q[:, 0] * q[:, 2] + q[:, 1] * q[:, 3])
            theta = np.arccos(np.clip(q[:, 3] ** 2 - q[:, 0] ** 2 - q[:, 1] ** 2 + q[:, 2] ** 2, -1.0, 1.0))
            phi = np.arctan2(q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2], q[:, 1] * q[:, 3] - q[:, 0] * q[:, 2])
            euler_type = 2
        elif p_to_axis_order == 123:
            psi = np.arctan2(2 * (q[:, 0] * q[:, 3] - q[:, 1] * q[:, 2]),
                             q[:, 3] ** 2 - q[:, 0] ** 2 - q[:, 1] ** 2 + q[:, 2] ** 2)
            theta = np.arcsin(2 * (q[:, 0] * q[:, 2] + q[:, 1] * q[:, 3]))
            phi = np.arctan2(2 * (q[:, 2] * q[:, 3] - q[:, 0] * q[:, 1]),
                             q[:, 3] ** 2 + q[:, 0] ** 2 - q[:, 1] ** 2 - q[:, 2] ** 2)
            euler_type = 1
        elif p_to_axis_order == 231:
            psi = np.arctan2(2 * (q[:, 1] * q[:, 3] - q[:, 0] * q[:, 2]),
                             q[:, 3] ** 2 + q[:, 0] ** 2 - q[:, 1] ** 2 - q[:, 2] ** 2)
            theta = np.arcsin(2 * (q[:, 0] * q[:, 1] + q[:, 2] * q[:, 3]))
            phi = np.arctan2(2 * (q[:, 0] * q[:, 3] - q[:, 2] * q[:, 1]),
                             q[:, 3] ** 2 - q[:, 0] ** 2 + q[:, 1] ** 2 - q[:, 2] ** 2)
            euler_type = 1
        elif p_to_axis_order == 312:
            psi = np.arctan2(2 * (q[:, 2] * q[:, 3] - q[:, 0] * q[:, 1]),
                             q[:, 3] ** 2 - q[:, 0] ** 2 + q[:, 1] ** 2 - q[:, 2] ** 2)
            theta = np.arcsin(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]))
            phi = np.arctan2(2 * (q[:, 1] * q[:, 3] - q[:, 2] * q[:, 0]),
                             q[:, 3] ** 2 - q[:, 0] ** 2 - q[:, 1] ** 2 + q[:, 2] ** 2)
            euler_type = 1
        elif p_to_axis_order == 132:
            psi = np.arctan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                             q[:, 3] ** 2 - q[:, 0] ** 2 + q[:, 1] ** 2 - q[:, 2] ** 2)
            theta = np.arcsin(2 * (q[:, 2] * q[:, 3] - q[:, 0] * q[:, 1]))
            phi = np.arctan2(2 * (q[:, 0] * q[:, 2] + q[:, 1] * q[:, 3]),
                             q[:, 3] ** 2 + q[:, 0] ** 2 - q[:, 1] ** 2 - q[:, 2] ** 2)
            euler_type = 1
        elif p_to_axis_order == 213:
            psi = np.arctan2(2 * (q[:, 0] * q[:, 2] + q[:, 1] * q[:, 3]),
                             q[:, 3] ** 2 - q[:, 0] ** 2 - q[:, 1] ** 2 + q[:, 2] ** 2)
            theta = np.arcsin(2 * (q[:, 0] * q[:, 3] - q[:, 1] * q[:, 2]))
            phi = np.arctan2(2 * (q[:, 0] * q[:, 1] + q[:, 2] * q[:, 3]),
                             q[:, 3] ** 2 - q[:, 0] ** 2 + q[:, 1] ** 2 - q[:, 2] ** 2)
            euler_type = 1
        elif p_to_axis_order == 321:
            psi = np.arctan2(2 * (q[:, 0] * q[:, 1] + q[:, 2] * q[:, 3]),
                             q[:, 3] ** 2 + q[:, 0] ** 2 - q[:, 1] ** 2 - q[:, 2] ** 2)
            theta = np.arcsin(2 * (q[:, 1] * q[:, 3] - q[:, 0] * q[:, 2]))
            phi = np.arctan2(2 * (q[:, 0] * q[:, 3] + q[:, 2] * q[:, 1]),
                             q[:, 3] ** 2 - q[:, 0] ** 2 - q[:, 1] ** 2 + q[:, 2] ** 2)
            euler_type = 1
        else:
            raise ValueError('Invalid output Euler angle order type (p_to).')

        if units == 'deg':
            output = np.mod(np.asarray([psi, theta, phi]) * 180. / np.pi, 360).T
        else:
            output = np.mod(np.asarray([psi, theta, phi]), 2 * np.pi).T

        output = output.reshape(in_shape + (3,))

        if perform_checks:
            if euler_type == 1:
                sing_chk = np.abs(theta) > (np.pi / 2 - 0.0017)  # 89.9 deg
                #sing_chk=sing_chk[sing_chk>0]
                if sing_chk:
                    print ('Exception: Input rotation %s s resides too close to' \
                           'Type 1 Euler singularity.\nType 1 ' \
                           'Euler singularity occurs when second ' \
                           'angle is -90 or 90 degrees.\nPlease choose ' \
                           'different output type.' % str(sing_chk))

            elif euler_type == 2:
                # TODO
                print "Euler Type 2 Singularity Check not implemented"
                #sing_chk=[find(np.abs(theta*180/np.pi)<0.1)find(np.abs(theta*180/np.pi-180)<0.1)find(np.abs(theta*180/np.pi-360))<0.1]
                #sing_chk=sort(sing_chk(sing_chk>0))
                #if size(sing_chk,1)>=1:
                #    print ('Exception: Input rotation ## s resides too close to Type 2 Euler np.singularity.\nType 2 Euler np.singularity occurs when second angle is 0 or 180 degrees.\nPlease choose different output type.',str(sing_chk(1,1)))

    return output


cdef rotation_matrix_to_quaternion(np.ndarray[FLOAT_TYPE_t, ndim=3] g_in):
    """
    Convert an array of 3x3 rotation matrices to an array of quaternions.

    :param g_in: a numpy array of shape (N, 3, 3), where N is the number of rotation matrices.
    :return: a numpy array of shape (N, 4) containing quaternions.
    """

    cdef int N = g_in.shape[0]
    cdef np.ndarray[FLOAT_TYPE_t, ndim=2] q = np.empty((N, 4))
    cdef np.ndarray[FLOAT_TYPE_t, ndim=2] denom = np.empty_like(q)
    denom[: ,0] = 0.5 * np.sqrt(np.abs(1 + g_in[:, 0, 0] - g_in[:, 1, 1] - g_in[:, 2, 2]))
    denom[:, 1] = 0.5 * np.sqrt(np.abs(1 - g_in[:, 0, 0] + g_in[:, 1, 1] - g_in[:, 2, 2]))
    denom[:, 2] = 0.5 * np.sqrt(np.abs(1 - g_in[:, 0, 0] - g_in[:, 1, 1] + g_in[:, 2, 2]))
    denom[:, 3] = 0.5 * np.sqrt(np.abs(1 + g_in[:, 0, 0] + g_in[:, 1, 1] + g_in[:, 2, 2]))

    case = np.argmax(denom, axis=1)

    for i in range(N):
        if case[i] == 0:
            q[i, 0] = denom[i, 0]
            q[i, 1] = (g_in[i, 0, 1] + g_in[i, 1, 0]) / (4 * q[i, 0])
            q[i, 2] = (g_in[i, 0, 2] + g_in[i, 2, 0]) / (4 * q[i, 0])
            q[i, 3] = (g_in[i, 1, 2] - g_in[i, 2, 1]) / (4 * q[i, 0])
        elif case[i] == 1:
            q[i, 1] = denom[i, 1]
            q[i, 0] = (g_in[i, 0, 1] + g_in[i, 1, 0]) / (4 * q[i, 1])
            q[i, 2] = (g_in[i, 1, 2] + g_in[i, 2, 1]) / (4 * q[i, 1])
            q[i, 3] = (g_in[i, 2, 0] - g_in[i, 0, 2]) / (4 * q[i, 1])
        elif case[i] == 2:
            q[i, 2] = denom[i, 2]
            q[i, 0] = (g_in[i, 0, 2] + g_in[i, 2, 0]) / (4 * q[i, 2])
            q[i, 1] = (g_in[i, 1, 2] + g_in[i, 2, 1]) / (4 * q[i, 2])
            q[i, 3] = (g_in[i, 0, 1] - g_in[i, 1, 0]) / (4 * q[i, 2])
        elif case[i] == 3:
            q[i, 3] = denom[i, 3]
            q[i, 0] = (g_in[i, 1, 2] - g_in[i, 2, 1]) / (4 * q[i, 3])
            q[i, 1] = (g_in[i, 2, 0] - g_in[i, 0, 2]) / (4 * q[i, 3])
            q[i, 2] = (g_in[i, 0, 1] - g_in[i, 1, 0]) / (4 * q[i, 3])

    return q


# The following was found in licence.txt for spincalc.m,
# from on which the change_parameterization function in this file is derived:
#
# Copyright (c) 2011, John Fuller
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in
#       the documentation and/or other materials provided with the distribution
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import numpy as np
from lie_learn.groups.SO3 import *


def test_change_parameterization():

    def is_equal(R1, R2, p):
        if p == 'Q':
            # Quaternions are only defined up to a sign, so check each row, what sign we need
            for i in range(R1.shape[0]):
                if not (np.allclose(R1[i, ...], R2[i, ...]) or np.allclose(R1[i, ...], -R2[i, ...])):
                    return False
            return True
        elif p == 'EV':
            # Euler vector (x,y,z,theta) == (-x,-y,-z,-theta mod 2pi)
            for i in range(R1.shape[0]):
                R2i = np.array([-R2[i, 0], -R2[i, 1], -R2[i, 2], (-R2[i, 3]) % (2 * np.pi)])
                if not (np.allclose(R1[i, ...], R2[i, ...]) or np.allclose(R1[i, ], R2i)):
                    return False
            return True

        else:
            return np.allclose(R1, R2)

    for p1 in parameterizations:
        for p2 in parameterizations:

            # Create two random rotations in 313 Euler angles
            R1_EA313 = (np.random.rand(3) * np.array([2 * np.pi, np.pi, 2 * np.pi]))[np.newaxis, :]
            R2_EA313 = (np.random.rand(3) * np.array([2 * np.pi, np.pi, 2 * np.pi]))[np.newaxis, :]
            R_EA313 = np.r_[R1_EA313, R2_EA313]

            R1_p1 = change_coordinates(p_from='EA313', p_to=p1, g=R1_EA313)
            R1_p2 = change_coordinates(p_from='EA313', p_to=p2, g=R1_EA313)
            R2_p1 = change_coordinates(p_from='EA313', p_to=p1, g=R2_EA313)
            R2_p2 = change_coordinates(p_from='EA313', p_to=p2, g=R2_EA313)
            R_p1 = change_coordinates(p_from='EA313', p_to=p1, g=R_EA313)
            R_p2 = change_coordinates(p_from='EA313', p_to=p2, g=R_EA313)

            R1_p2_from_R1_p1 = change_coordinates(p_from=p1, p_to=p2, g=R1_p1)
            R1_p1_from_R1_p2 = change_coordinates(p_from=p2, p_to=p1, g=R1_p2)
            R2_p2_from_R2_p1 = change_coordinates(p_from=p1, p_to=p2, g=R2_p1)
            R2_p1_from_R2_p2 = change_coordinates(p_from=p2, p_to=p1, g=R2_p2)
            R_p2_from_R_p1 = change_coordinates(p_from=p1, p_to=p2, g=R_p1)
            R_p1_from_R_p2 = change_coordinates(p_from=p2, p_to=p1, g=R_p2)

            assert is_equal(R1_p1_from_R1_p2, R1_p1, p1), (
                p1 + ' to ' + p2 + ' | R1_p1: ' + str(R1_p1) + ' | R1_p2: ' + str(R1_p2) + ' | R1_p1_from_R1_p2: ' +
                str(R1_p1_from_R1_p2))
            assert is_equal(R2_p1_from_R2_p2, R2_p1, p1), (
                p1 + ' to ' + p2 + ' | R2_p1: ' + str(R2_p1) + ' | R2_p2: ' + str(R2_p2) + ' | R2_p1_from_R2_p2: ' +
                str(R2_p1_from_R2_p2))
            assert is_equal(R_p1_from_R_p2, R_p1, p1), (
                p1 + ' to ' + p2 + ' | R_p1: ' + str(R_p1) + ' | R_p2: ' + str(R_p2) + ' | R_p1_from_R_p2: ' +
                str(R_p1_from_R_p2))
            assert is_equal(R1_p2_from_R1_p1, R1_p2, p2), (
                p1 + ' to ' + p2 + ' | R1_p1: ' + str(R1_p1) + ' | R1_p2: ' + str(R1_p2) + ' | R1_p2_from_R1_p1: ' +
                str(R1_p2_from_R1_p1))
            assert is_equal(R2_p2_from_R2_p1, R2_p2, p2), (
                p1 + ' to ' + p2 + ' | R2_p1: ' + str(R2_p1) + ' | R2_p2: ' + str(R2_p2) + ' | R2_p2_from_R2_p1: ' +
                str(R2_p2_from_R2_p1))
            assert is_equal(R_p2_from_R_p1, R_p2, p2), (
                p1 + ' to ' + p2 + ' | R_p1: ' + str(R_p1) + ' | R_p2: ' + str(R_p2) + ' | R_p2_from_R_p1: ' +
                str(R_p2_from_R_p1))

def test_invert():

    for p in parameterizations:

        R_EA = np.random.rand(4, 5, 6, 3) * np.array([2 * np.pi, np.pi, 2 * np.pi])[None, None, None, :]
        R_p = change_coordinates(R_EA, p_from='EA313', p_to=p)
        R_p_inv = invert(R_p, parameterization=p)

        e = compose(R_p, R_p_inv, parameterization=p)
        eM = change_coordinates(e, p_from=p, p_to='MAT')
        assert np.isclose(np.sum(eM - np.eye(3)), 0.0), 'not the identity: ' + eM
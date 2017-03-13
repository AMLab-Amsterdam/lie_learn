from lie_learn.representations.SO3.irrep_bases import *
from .spherical_harmonics import *

TEST_L_MAX = 5

def test_change_of_basis_matrix():
    """
    Testing if change of basis matrix is consistent with spherical harmonics functions
    """

    for l in range(TEST_L_MAX):
        theta = np.random.rand() * np.pi
        phi = np.random.rand() * np.pi * 2
        for from_field in ['complex', 'real']:
            for from_normalization in ['seismology', 'quantum', 'geodesy', 'nfft']:
                for from_cs in ['cs', 'nocs']:
                    for to_field in ['complex', 'real']:
                        for to_normalization in ['seismology', 'quantum', 'geodesy', 'nfft']:
                            for to_cs in ['cs', 'nocs']:
                                Y_from = sh(l, np.arange(-l, l + 1), theta, phi,
                                            from_field, from_normalization, from_cs == 'cs')

                                Y_to = sh(l, np.arange(-l, l + 1), theta, phi,
                                          to_field, to_normalization, to_cs == 'cs')

                                B = change_of_basis_matrix(l=l,
                                                           frm=(from_field, from_normalization, 'centered', from_cs),
                                                           to=(to_field, to_normalization, 'centered', to_cs))

                                print(from_field, from_normalization, from_cs, '->', to_field, to_normalization, to_cs, np.sum(np.abs(B.dot(Y_from) - Y_to)))
                                assert np.isclose(np.sum(np.abs(B.dot(Y_from) - Y_to)), 0.0)
                                assert np.isclose(np.sum(np.abs(np.linalg.inv(B).dot(Y_to) - Y_from)), 0.0)


def test_change_of_basis_function():
    """
    Testing if change of basis function is consistent with spherical harmonics functions
    """

    for l in range(TEST_L_MAX):
        theta = np.random.rand() * np.pi
        phi = np.random.rand() * np.pi * 2
        for from_field in ['complex', 'real']:
            for from_normalization in ['seismology', 'quantum', 'geodesy', 'nfft']:
                for from_cs in ['cs', 'nocs']:
                    for to_field in ['complex', 'real']:
                        for to_normalization in ['seismology', 'quantum', 'geodesy', 'nfft']:
                            for to_cs in ['cs', 'nocs']:

                                Y_from = sh(l, np.arange(-l, l + 1), theta, phi,
                                            from_field, from_normalization, from_cs == 'cs')

                                Y_to = sh(l, np.arange(-l, l + 1), theta, phi,
                                          to_field, to_normalization, to_cs == 'cs')

                                f = change_of_basis_function(l=l,
                                                             frm=(from_field, from_normalization, 'centered', from_cs),
                                                             to=(to_field, to_normalization, 'centered', to_cs))

                                print(from_field, from_normalization, from_cs, '->', to_field, to_normalization, to_cs, np.sum(np.abs(f(Y_from) - Y_to)))
                                assert np.isclose(np.sum(np.abs(f(Y_from) - Y_to)), 0.0)


def test_change_of_basis_function_lists():
    """
    Testing change of basis function for spherical harmonics for multiple orders at once.
    The change-of-basis function for spherical harmonics should be consistent with the CSH & RSH functions.
    """
    l = np.arange(4)
    ls = np.array([0, 1,1,1, 2,2,2,2,2, 3,3,3,3,3,3,3])
    ms = np.array([0, -1,0,1, -2,-1,0,1,2, -3,-2,-1,0,1,2,3])

    theta = np.random.rand() * np.pi
    phi = np.random.rand() * np.pi * 2
    for from_field in ['complex', 'real']:
        for from_normalization in ['seismology', 'quantum', 'geodesy', 'nfft']:
            for from_cs in ['cs', 'nocs']:
                for to_field in ['complex', 'real']:
                    for to_normalization in ['seismology', 'quantum', 'geodesy', 'nfft']:
                        for to_cs in ['cs', 'nocs']:

                            Y_from = sh(ls, ms, theta, phi,
                                        from_field, from_normalization, from_cs == 'cs')

                            Y_to = sh(ls, ms, theta, phi,
                                      to_field, to_normalization, to_cs == 'cs')

                            f = change_of_basis_function(l=l,
                                                         frm=(from_field, from_normalization, 'centered', from_cs),
                                                         to=(to_field, to_normalization, 'centered', to_cs))

                            print(from_field, from_normalization, from_cs, '->', to_field, to_normalization, to_cs, np.sum(np.abs(f(Y_from) - Y_to)))
                            assert np.isclose(np.sum(np.abs(f(Y_from) - Y_to)), 0.0)


def test_invertibility():
    """
    Testing if change_of_basis_function for SO(3) is invertible
    """

    for l in range(TEST_L_MAX):
        theta = np.random.rand() * np.pi
        phi = np.random.rand() * np.pi * 2
        for from_field in ['complex', 'real']:
            for from_normalization in ['seismology', 'quantum', 'geodesy', 'nfft']:
                for from_cs in ['cs', 'nocs']:
                    for from_order in ['centered', 'block']:
                        for to_field in ['complex', 'real']:
                            for to_normalization in ['seismology', 'quantum', 'geodesy', 'nfft']:
                                for to_cs in ['cs', 'nocs']:
                                    for to_order in ['centered', 'block']:
                                        # A truly complex function cannot be made real;
                                        if from_field == 'complex' and to_field == 'real':
                                            continue

                                        if from_field == 'complex':
                                            Y = np.random.randn(2 * l + 1) + np.random.randn(2 * l + 1) * 1j
                                        else:
                                            Y = np.random.randn(2 * l + 1)

                                        f = change_of_basis_function(l=l,
                                                                     frm=(from_field, from_normalization, from_order, from_cs),
                                                                     to=(to_field, to_normalization, to_order, to_cs))

                                        f_inv = change_of_basis_function(l=l,
                                                                         frm=(to_field, to_normalization, to_order, to_cs),
                                                                         to=(from_field, from_normalization, from_order, from_cs))


                                        print(from_field, from_normalization, from_cs, from_order, '->', to_field, to_normalization, to_cs, to_order, np.sum(np.abs(f_inv(f(Y)) - Y)))
                                        assert np.isclose(np.sum(np.abs(f_inv(f(Y)) - Y)), 0.)
                                        #assert np.isclose(np.sum(np.abs(f(f_inv(Y)) - Y)), 0.)


def test_linearity_change_of_basis():
    """
    Testing that SO3 change of basis is indeed linear
    """
    for l in range(TEST_L_MAX):
        theta = np.random.rand() * np.pi
        phi = np.random.rand() * np.pi * 2
        for from_field in ['complex', 'real']:
            for from_normalization in ['seismology', 'quantum', 'geodesy', 'nfft']:
                for from_cs in ['cs', 'nocs']:
                    for from_order in ['centered', 'block']:
                        for to_field in ['complex', 'real']:
                            for to_normalization in ['seismology', 'quantum', 'geodesy', 'nfft']:
                                for to_cs in ['cs', 'nocs']:
                                    for to_order in ['centered', 'block']:

                                        # A truly complex function cannot be made real;
                                        if from_field == 'complex' and to_field == 'real':
                                            continue

                                        Y1 = np.random.randn(2 * l + 1)
                                        Y2 = np.random.randn(2 * l + 1)
                                        a = np.random.randn(1)
                                        b = np.random.randn(1)

                                        f = change_of_basis_function(l=l,
                                                                     frm=(from_field, from_normalization, from_order, from_cs),

                                                                     to=(to_field, to_normalization, from_order, to_cs))

                                        print(from_field, from_normalization, from_cs, from_order, '->', to_field, to_normalization, to_cs, to_order, np.sum(np.abs(a * f(Y1) + b * f(Y2) - f(a*Y1 + b*Y2))))
                                        assert np.isclose(np.sum(np.abs(a * f(Y1) + b * f(Y2) - f(a*Y1 + b*Y2))), 0.)

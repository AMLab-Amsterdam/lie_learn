

import numpy as np
from scipy.fftpack import rfft, irfft
from scipy.optimize import fmin_l_bfgs_b

"""
samples = vonmises.rvs(kappa=1.0, size=10000)
J = 10; alpha = 10; n = (2 * J + 1) * alpha
even = np.arange(0, 2 * J, 2); odd = np.arange(1, 2 * J, 2)
empirical_moments = np.zeros(2 * J)
empirical_moments[even] = np.mean(np.cos(np.arange(1, J + 1)[np.newaxis, :] * samples[:, np.newaxis]), axis=0)
empirical_moments[odd] = np.mean(np.sin(np.arange(1, J + 1)[np.newaxis, :] * samples[:, np.newaxis]), axis=0)


def logp_and_grad(eta):
    # Compute moments:
    negative_energy = irfft(np.hstack([[0], eta]), n=n) * (n / 2.)
    unnormalized_moments = rfft(np.exp(negative_energy)) / (n / 2.) * np.pi
    Z = unnormalized_moments[0]
    moments = unnormalized_moments[1:eta.size + 1] / Z

    # Compute gradients and log-prob:
    grad_logp = empirical_moments - moments
    logp = eta.dot(empirical_moments) - np.log(Z)
    return -logp, -grad_logp

opt_eta, opt_neg_logp, info = fmin_l_bfgs_b(logp_and_grad, x0=np.zeros(2 * J), iprint=0, factr=1e7, pgtol=1e-5)
print info['task']
print 'Optimum log-likelihood:', -opt_neg_logp
print 'Optimal parameters:', np.round(opt_eta, 2)
"""


class S2HarmonicDensity():

    def __init__(self, L_max, oversampling_factor=2):

        self.L_max = L_max
        self.L_max_os = self.L_max * oversampling_factor

        self.even = np.arange(0, 2 * L_max, 2)
        self.odd = np.arange(1, 2 * L_max, 2)

        self.even_os = np.arange(0, 2 * self.L_max_os, 2)
        self.odd_os = np.arange(1, 2 * self.L_max_os, 2)


    def negative_energy(self, x, eta):
        """

        :param x:
        :param eta:
        :return:
        """

        pass #return eta.dot(sh(self.ls[:, np.newaxis], self.ms[:, np.newaxis],
             #             x[:, 0][np.newaxis, :], x[:, 1][np.newaxis, :],
             #             field='real', normalization='quantum', condon_shortley=True))



    def moments(self, eta):
        """

        :param eta:
        :return:
        """
        pass

    def empirical_moments(self, X, average=True):
        """
        Compute the empirical moments of the sample x

        :param x: dataset shape (N, 2) for 2 spherical coordinates (theta, phi) per point
        :return: the moments 1/N sum_i=1^N T(x_i)
        """
        pass

    def grad_log_p(self, eta, empirical_moments):
        """

        :param eta:
        :param M:
        :return:
        """
        pass

    def log_p_and_grad(self, eta, empirical_moments):
        """

        """
        pass


    def mle_lbfgs(self, empirical_moments, eta_init=None, SigmaInv=None, verbose=True):

        # Move to base-class?

        if eta_init is None:
            eta = np.zeros((self.L_max + 1) ** 2 - 1)
        else:
            eta = eta_init.copy()

        if SigmaInv is None:  # No regularization
            def objective_and_grad(eta):
                logp, grad = self.log_p_and_grad(eta, empirical_moments)
                return -logp, -grad
        else:
            lnZ_prior = 0.5 * SigmaInv.size * np.log(2 * np.pi) - 0.5 * np.sum(np.log(SigmaInv))
            def objective_and_grad(eta):
                logp, grad = self.log_p_and_grad(eta, empirical_moments)
                SigmaInv_eta = SigmaInv * eta
                logp += -0.5 * eta.dot(SigmaInv_eta) - lnZ_prior
                grad += -SigmaInv_eta
                return -logp, -grad

        opt_eta, opt_neg_logp, info = fmin_l_bfgs_b(objective_and_grad, x0=eta, iprint=int(verbose) - 1,
                                                    factr=1e7,  # moderate accuracy
                                                    #factr=1e12,  # low accuracy
                                                    pgtol=1e-5)  # norm of proj. grad. to stop iteration at

        if verbose:
            print('Maximum log prob:', -opt_neg_logp)
            print('Optimization info:', info)

        # Finally, compute Z:
        _, lnZ = self.moments(opt_eta)
        return opt_eta, lnZ


    def _moment_numerical_integration(self, eta, l, m):
        """
        Compute the (l,m)-moment of the density with natural parameter eta using slow numerical integration.
        The output of this function should be equal to the *unnormalized* moment as it comes out of the FFT
        (without dividing by Z).

        :param eta:
        :param l:
        :param m:
        :return:
        """
        pass
        #f = lambda theta, phi: (np.exp(self.negative_energy(np.array([[theta, phi]]), eta))
        #                        * sh(l, m, theta, phi,
        #                             field='real', normalization='quantum',
        #                             condon_shortley=True))
        #return S2.integrate(f)


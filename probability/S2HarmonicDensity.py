
import numpy as np
from scipy.optimize import fmin_cg, fmin_l_bfgs_b
from ..spectral.S2FFT_NFFT import S2FFT_NFFT
from ..spaces import S2
from ..representations.SO3.spherical_harmonics import sh


# TODO: add unit-test to check moments agains numerical integration (have done this in terminal)

class S2HarmonicDensity():

    def __init__(self, L_max, oversampling_factor=2, fft=None):

        # Compute the maximum degree from the length of eta
        # sum_l=0^L (2 l + 1) = (L + 1)^2
        # so
        # L = sqrt(eta.size) - 1
        #if (np.sqrt(eta.shape[-1] + 1) != np.sqrt(eta.shape[-1] + 1).astype(int)).any():
        #    raise ValueError('Incorrect eta: last dimension must be a square.')
        #self.L_max = int(np.sqrt(eta.shape[-1] + 1) - 1)
        self.L_max = L_max
        self.L_max_os = self.L_max * oversampling_factor

        # Create arrays containing the (l,m) coordinates
        l = [[l] * (2 * l + 1) for l in range(1, self.L_max + 1)]
        self.ls = np.array([ll for sublist in l for ll in sublist])  # 1, 1, 1, 2, 2, 2, 2, 2, ...
        l_oversampled = [[l] * (2 * l + 1) for l in range(1, self.L_max_os + 1)]
        self.ls_oversampled = np.array([ll for sublist in l_oversampled for ll in sublist])

        m = [list(range(-l, l + 1)) for l in range(1, self.L_max + 1)]
        self.ms = np.array([mm for sublist in m for mm in sublist])  # -1, 0, 1, -2, -1, 0, 1, 2, ...
        m_oversampled = [list(range(-l, l + 1)) for l in range(1, self.L_max_os + 1)]
        self.ms_oversampled = np.array([mm for sublist in m_oversampled for mm in sublist])

        if fft is None:
            # Setup a spherical grid and corresponding quadrature weights
            convention = 'Clenshaw-Curtis'
            #convention = 'Gauss-Legendre'

            x = S2.meshgrid(b=self.L_max_os, grid_type=convention)
            w = S2.quadrature_weights(b=self.L_max_os, grid_type=convention)
            self.fft = S2FFT_NFFT(L_max=self.L_max_os, x=x, w=w)
        else:
            if fft.L_max < self.L_max_os:
                raise ValueError('fft.L_max must be larger than or equal to L_max * oversampling_factor')
            self.fft = fft

    def negative_energy(self, x, eta):
        """

        :param x:
        :param eta:
        :return:
        """
        x = np.atleast_2d(x)
        return eta.dot(sh(self.ls[:, np.newaxis], self.ms[:, np.newaxis],
                          x[:, 0][np.newaxis, :], x[:, 1][np.newaxis, :],
                          field='real', normalization='quantum', condon_shortley=True))

    def sufficient_statistics(self, x):
        x = np.atleast_2d(x)
        return sh(self.ls[:, np.newaxis], self.ms[:, np.newaxis],
                  x[:, 0][np.newaxis, :], x[:, 1][np.newaxis, :],
                  field='real', normalization='quantum', condon_shortley=True)

    def moments(self, eta):
        """

        :param eta:
        :return:
        """
        #
        eta_os = np.zeros((self.L_max_os + 1) ** 2)
        eta_os[1:eta.size + 1] = eta

        neg_e = self.fft.synthesize(eta_os)
        #unnormalized_moments = self.fft.analyze(np.exp(neg_e))
        #return unnormalized_moments[1:] / unnormalized_moments[0], unnormalized_moments[0]

        maximum = np.max(neg_e)
        unnormalized_moments = self.fft.analyze(np.exp(neg_e - maximum))

        #log_unnormalized_moments = np.log(unnormalized_moments + 0j)
        #moments = np.exp(log_unnormalized_moments - log_unnormalized_moments[0]).real
        #Z = np.exp(log_raw_moments[0] + maximum).real
        #lnZ = (log_unnormalized_moments[0] + maximum).real

        unnormalized_moments[0] *= np.sqrt(4 * np.pi)

        moments = unnormalized_moments / unnormalized_moments[0]
        #print np.sum(np.abs(m2 - moments))
        lnZ = np.log(unnormalized_moments[0]) + maximum
        #print lnZ, lnZ2, lnZ - lnZ2

        return moments[1:(self.L_max + 1) ** 2], lnZ

    def moments_numint(self, eta):

        moments = np.zeros((self.L_max + 1) ** 2)

        f = lambda th, ph: np.exp(self.negative_energy([th, ph], eta))
        moments[0] = S2.integrate(f, normalize=False)

        for l in range(1, self.L_max + 1):
            for m in range(-l, l + 1):
                print('integrating', l, m)
                f = lambda th, ph: np.exp(self.negative_energy([th, ph], eta)) * sh(l, m, th, ph,
                                                        field='real', normalization='quantum', condon_shortley=True)
                moments[l ** 2 + l + m] = S2.integrate(f, normalize=False)

        return moments[1:] / moments[0], moments[0]

    def empirical_moments(self, X, average=True):
        """
        Compute the empirical moments of the sample x

        :param x: dataset shape (N, 2) for 2 spherical coordinates (theta, phi) per point
        :return: the moments 1/N sum_i=1^N T(x_i)
        """
        # TODO: this can be done potentially more efficiently by computing T(0,0) (the suff. stats. of the north pole),
        # and then transforming by D(theta, phi, 0) or something similar. This matrix vector-multiplication can be
        # done efficiently by the Pinchon-Hoggan method. (or asymptotically even faster using other methods)

        T = sh(self.ls[np.newaxis, :], self.ms[np.newaxis, :],
               X[:, 0][:, np.newaxis], X[:, 1][:, np.newaxis],
               field='real', normalization='quantum', condon_shortley=True)
        if average:
            return T.mean(axis=0)
        else:
            return T

    def grad_log_p(self, eta, empirical_moments):
        """

        :param eta:
        :param M:
        :return:
        """
        moments, _ = self.moments(eta)
        return empirical_moments - moments

    def log_p_and_grad(self, eta, empirical_moments):
        """
        Compute the gradient of the log probability of the density given by eta,
        evaluated at a sample of data summarized by the empirical moments.
        The log-prob is:
        ln prod_i=1^N p(x_i | eta)
         =
        sum_i=1^N eta^T T(x_i) - ln Z_eta
         =
        N (eta^T T_bar - ln Z_eta)
        where T_bar = 1/N sum_i=1^N T(x_i) are the empirical moments, as computed by self.empirical_moments(X).
        In this function we work with the *average* log-prob, i.e. leaving out the factor N from the log-prob formula.

        The gradient is (leaving out the factor of N)
        T_bar - E_eta[T(x)]
        where E_eta[T(x)] are the moments of p(x|eta), as computed by self.moments(eta).

        :param eta: the natural parameters of the distribution
        :param empirical_moments: the average sufficient statistics, as computed by self.empirical_moments(X)
        :return: the gradient of the average log-prob with respect to eta, and the average log prob itself.
        """
        moments, lnZ = self.moments(eta)
        grad_logp = empirical_moments - moments
        logp = eta.dot(empirical_moments) - lnZ
        return logp, grad_logp

    def mle_sgd(self, empirical_moments, eta_init=None, learning_rate=0.1, max_iter=1000, verbose=True):
        """

        :param X:
        :return:
        """
        if eta_init is None:
            eta = np.zeros((self.L_max + 1) ** 2 - 1)
        else:
            eta = eta_init.copy()

        for i in range(max_iter):
            log_p, grad_log_p = self.log_p_and_grad(eta, empirical_moments)
            eta += learning_rate * grad_log_p
            if verbose:
                print('log-prob:', log_p)

        # Finally, compute Z:
        _, lnZ = self.moments(eta)
        return eta, lnZ

    def mle_lbfgs(self, empirical_moments, eta_init=None, SigmaInv=None, verbose=True):

        if eta_init is None:
            eta = np.zeros((self.L_max + 1) ** 2 - 1)
        else:
            eta = eta_init.copy()

        if SigmaInv is None:  # No regularization
            def objective_and_grad(eta):
                logp, grad = self.log_p_and_grad(eta, empirical_moments)
                return -logp, -grad
        else:
            #lnZ_prior = 0.5 * SigmaInv.size * np.log(2 * np.pi) - 0.5 * np.sum(np.log(SigmaInv))
            def objective_and_grad(eta):
                logp, grad = self.log_p_and_grad(eta, empirical_moments)
                SigmaInv_eta = SigmaInv * eta
                logp += -0.5 * eta.dot(SigmaInv_eta)  # - lnZ_prior
                grad += -SigmaInv_eta
                return -logp, -grad

        opt_eta, opt_neg_logp, info = fmin_l_bfgs_b(objective_and_grad, x0=eta, iprint=int(verbose) - 1,
                                                    #factr=1e7,  # moderate accuracy
                                                    factr=1e12,  # low accuracy
                                                    pgtol=1e-4,
                                                    maxiter=1000)  # norm of proj. grad. to stop iteration at

        if verbose:
            print('Maximum log prob:', -opt_neg_logp)
            print('Optimization info:', info['warnflag'], info['task'], np.mean(info['grad']))

        # Finally, compute Z:
        _, lnZ = self.moments(opt_eta)
        return opt_eta, lnZ

    def mle_cg(self, empirical_moments, eta_init=None, verbose=True):

        if eta_init is None:
            eta = np.zeros((self.L_max + 1) ** 2 - 1)
        else:
            eta = eta_init.copy()

        def objective(eta):
            logp, _ = self.log_p_and_grad(eta, empirical_moments)
            return -logp
        def grad(eta):
            _, grad = self.log_p_and_grad(eta, empirical_moments)
            return -grad
        eta_min, logp_min, fun_calls, grad_calls, warnflag = fmin_cg(f=objective, fprime=grad, x0=eta,
                                                                     full_output=True)

        if verbose:
            print('min log p:', logp_min)
            print('fun_calls:', fun_calls)
            print('grad_calls:', grad_calls)
            print('warnflag:', warnflag)
            #print 'allvecs:', allvecs

        # Finally, compute Z:
        _, lnZ = self.moments(eta_min)
        return eta_min, lnZ

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
        f = lambda theta, phi: (np.exp(self.negative_energy(np.array([[theta, phi]]), eta))
                                * sh(l, m, theta, phi,
                                     field='real', normalization='quantum',
                                     condon_shortley=True))
        return S2.integrate(f)

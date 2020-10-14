import numpy as np
from scipy.stats import multivariate_normal

def sample_from_k_theta_1d(mean, sd, nb_samples):
    return np.random.normal(mean, sd, nb_samples)


class AlphaGammaDescent:

    def __init__(self, i, lnprob, D, thetas_init, alpha, T, N, J_t, h_t, nb_samples_y_t, eta_n, func_eval,
                 bool_md, bool_average, increment, freq_eval):
        '''
        Initialise parameters in the AlphaGammaDescent algorithm
        :param i: used for paralellisation (int)
        :param lnprob: function which computes the log prob of a given model (function)
        :param D: dimension of the latent space (int)
        :param thetas_init: initial set (theta_1, ..., theta_J_t)
        :param alpha: parameter of the alpha-divergence (float)
        :param T: number of overall iterations T (int)
        :param N: number of iterations in the AlphaGammaDescent (int)
        :param J_t: number of centers (int)
        :param h_t: bandwidth (float > 0)
        :param nb_samples_y_t: number of samples used to estimate bmu (int)
        :param eta_n: learning rate (float > 0)
        :param func_eval: compute the accuracy (function)
        :param bool_md: if True, performs Mirror Descent. if False, performs Power Descent (boolean)
        :param bool_average: if True, consider the average lambda_N = 1/N sum_n lambda_n
                             if False, consider only the value at time N (boolean)
        :param increment: increment for the number of centers J_t+1 = J_t + increment (int)
        :param freq_eval: frequence at which the function func_eval is called
        '''
        # Model
        self.lnprob = lnprob
        self.D = D

        # Initial thetas
        self.thetas_init = thetas_init.copy()

        # Parameters
        self.alpha = alpha
        self.T = T
        self.N = N
        self.J_t = J_t
        self.h_t = h_t
        self.nb_samples_y_t = nb_samples_y_t

        self.eta_n = eta_n
        self.eta_n0 = eta_n

        if self.D == 1:
            self._generate_from_multivariate = sample_from_k_theta_1d
        else:
            self._generate_from_multivariate = self._sample_from_k_theta_nd
        self.func_eval = func_eval

        self.bool_md = bool_md
        self.bool_average = bool_average
        self.increment = increment
        self.freq_eval = freq_eval

    def _sample_from_k_theta_nd(self, mean, sd, nb_samples):
        return np.random.multivariate_normal(mean=mean, cov=sd * np.identity(self.D), size=nb_samples)

    def _sample_from_muk(self, means, sd, weights, nb_samples_y, nb_centers):
        repartition = np.random.multinomial(nb_samples_y, weights)

        samples = []
        for i in range(nb_centers):
            nb = repartition[i]
            u = self._generate_from_multivariate(means[i], sd, nb)
            samples.extend(u)
        return np.array(samples)

    def _compute_bmu_one(self, weights, theta):
        bmu_one = np.zeros(self.J_t)
        repartition = np.random.multinomial(self.nb_samples_y_t, weights)
        llh = 0

        for j in range(self.J_t):
            nb = repartition[j]
            for _ in range(nb):
                y = self._generate_from_multivariate(theta[j], self.h_t, 1)[0]

                k_theta_y = multivariate_normal.pdf(theta, mean=y, cov=self.h_t * np.identity(self.D))
                muk_y = sum([weights[i] * k_theta_y[i] for i in range(self.J_t)])
                lnprob = self.lnprob(y)

                llh += np.exp(lnprob) / muk_y
                bmu_one += 1/muk_y * (np.log(muk_y) - lnprob) * k_theta_y

        llh = np.log(llh/self.nb_samples_y_t)

        return bmu_one, llh

    def _compute_propto_bmu_alpha(self, weights, theta):
        propto_bmu_alpha = np.zeros(self.J_t)
        llh = 0
        repartition = np.random.multinomial(self.nb_samples_y_t, weights)

        for j in range(self.J_t):
            nb = repartition[j]
            for _ in range(nb):
                y = self._generate_from_multivariate(theta[j], self.h_t, 1)[0]

                k_theta_y = multivariate_normal.pdf(theta, mean=y, cov=self.h_t * np.identity(self.D))
                muk_y = sum([weights[i] * k_theta_y[i] for i in range(self.J_t)])
                lnprob = self.lnprob(y)

                llh += np.exp(lnprob) / muk_y
                propto_bmu_alpha += 1 / muk_y * np.exp((self.alpha - 1) * (np.log(muk_y) - lnprob)) * k_theta_y

        llh = np.log(1 / self.nb_samples_y_t * llh)

        return propto_bmu_alpha, llh

    def _weights_update(self, weights, theta):

        if self.alpha == 1.:
            propto_bmu_alpha, llh = self._compute_bmu_one(weights, theta)
            argument = - self.eta_n / self.nb_samples_y_t * propto_bmu_alpha
            max_value = max(argument) #prevent overflow
            W = weights * np.exp(argument - max_value)

            renyi_bound = - 1 / self.nb_samples_y_t * np.sum(weights * propto_bmu_alpha)

        else:
            propto_bmu_alpha, llh = self._compute_propto_bmu_alpha(weights, theta)
            if self.bool_md:
                argument = self.eta_n / (self.nb_samples_y_t * (1 - self.alpha)) * propto_bmu_alpha
                max_value = max(argument)
                W = weights * np.exp(argument - max_value)
            else:
                W = weights * np.power(propto_bmu_alpha, self.eta_n / (1 - self.alpha))

            renyi_bound = 1 / (1 - self.alpha) * np.log(1 / self.nb_samples_y_t * np.sum(weights * propto_bmu_alpha))

        w = np.sum(W)

        return W / w, llh, renyi_bound

    def _compute_renyi_bound(self, weights, thetas):

        if self.alpha == 1.:
            bmu_alpha, llh = self._compute_bmu_one(weights, thetas)
            bound = - 1 / self.nb_samples_y_t * np.sum(weights * bmu_alpha)
        else:
            bmu_alpha, llh = self._compute_propto_bmu_alpha(weights, thetas)
            bound = 1 / (1 - self.alpha) * np.log(1 / self.nb_samples_y_t * np.sum(weights * bmu_alpha))

        return bound, llh

    def AlphaGammaDescent_algorithm(self, thetas, weights_init):

        self.eta_n = self.eta_n0
        weights_sum = weights_init.copy()
        weights, llh, renyi_bound = self._weights_update(weights_init, thetas)
        renyi_bound_lst = [renyi_bound]
        llh_lst = [llh]

        n_iter = 1
        weights_sum += weights.copy()

        while n_iter < self.N:
            self.eta_n = self.eta_n0 / (1 + np.sqrt(n_iter))
            weights, llh, renyi_bound = self._weights_update(weights, thetas)

            if self.bool_average:
                weights_sum += self.eta_n * weights.copy()

            llh_lst.append(llh)
            renyi_bound_lst.append(renyi_bound)
            n_iter += 1

        if self.bool_average:
            weights = weights_sum.copy()
            weights = weights / np.sum(weights)

        return weights, llh_lst, renyi_bound_lst

    def _thetas_update_sto_perturbation(self, weights, thetas, previous_J_t):
        thetas = self._sample_from_muk(thetas, self.h_t, weights, self.J_t, previous_J_t)
        return np.reshape(thetas, (-1, self.D))

    def _update_hyperparameters_as_in_pmd(self, t):
        previous_J_t = self.J_t
        self.J_t = self.J_t + self.increment
        self.nb_samples_y_t = self.J_t
        previous_h_t = self.h_t
        self.h_t = np.power(self.J_t, -1/(4 + self.D))
        self.eta_n = self.eta_n0
        return previous_J_t, previous_h_t

    def _full_algorithm(self):
        evaluation_lst = []
        thetas = self.thetas_init.copy()

        # Init with np.array([1 / self.J_t] * self.J_t)
        weights, llh, renyi_bound = self.AlphaGammaDescent_algorithm(thetas, np.array([1 / self.J_t] * self.J_t))

        renyi_bound_lst = renyi_bound
        llh_lst = llh
        evaluation = self.func_eval(weights, thetas, self.h_t, self.J_t)
        evaluation_lst.append(evaluation)

        previous_J_t, previous_h_t = self._update_hyperparameters_as_in_pmd(1)
        thetas = self._thetas_update_sto_perturbation(weights, thetas, previous_J_t)

        t = 1
        while t < self.T:
            print(t)
            # Init with np.array([1 / self.J_t] * self.J_t)
            weights, llh, renyi_bound = self.AlphaGammaDescent_algorithm(thetas, np.array([1 / self.J_t] * self.J_t))

            if t % self.freq_eval == 0:
                evaluation = self.func_eval(weights, thetas, self.h_t, self.J_t)
                evaluation_lst.append(evaluation)

            previous_J_t, previous_h_t = self._update_hyperparameters_as_in_pmd(t + 1)
            renyi_bound_lst.extend(renyi_bound)
            llh_lst.extend(llh)

            if t < self.T - 1:
                # Sample the new thetas
                thetas = self._thetas_update_sto_perturbation(weights, thetas, previous_J_t)

            t += 1

        evaluation = self.func_eval(weights, thetas, previous_h_t, previous_J_t)
        evaluation_lst.append(evaluation)

        return thetas, weights, renyi_bound_lst, llh_lst, evaluation_lst


import numpy as np
import scipy.io
from sklearn.model_selection import train_test_split
import numpy.matlib as nm
from scipy.stats import gamma, multivariate_normal
from scipy.io import loadmat
from functools import partial
from joblib import Parallel, delayed
from pathlib import Path

import AlphaGammaDescent as AGD

def sample_from_k_theta_1d(mean, sd, nb_samples):
    return np.random.normal(mean, sd, nb_samples)

'''
    Example of Bayesian Logistic Regression (the same setting as Gershman et al. 2012):
    The observed data D = {X, y} consist of N binary class labels, 
    y_t \in {-1,+1}, and d covariates for each datapoint, X_t \in R^d.
    The hidden variables \theta = {w, \alpha} consist of d regression coefficients w_k \in R,
    and a precision parameter \alpha \in R_+. We assume the following model:
        p(\alpha) = Gamma(\alpha; a, b)
        p(w_k | a) = N(w_k; 0, \alpha^-1)
        p(y_t = 1| x_t, w) = 1 / (1+exp(-w^T x_t))
    Code adapted from 
    https://github.com/DartML/Stein-Variational-Gradient-Descent/blob/master/python/bayesian_logistic_regression.py
'''

class BayesianLR:
    def __init__(self, X, Y, batchsize=100, a0=1, b0=0.01):
        self.X, self.Y = X, Y
        # TODO. Y in \in{+1, -1}
        self.batchsize = min(batchsize, X.shape[0])
        self.a0, self.b0 = a0, b0

        self.N0 = X.shape[0]
        self.permutation = np.random.permutation(self.N0)
        self.iter = 0

        self._generate_from_multivariate = self._sample_from_k_theta_nd

    def _sample_from_k_theta_nd(self, mean, sd, nb_samples, dim_latent):
        return np.random.multivariate_normal(mean=mean, cov=sd * np.identity(dim_latent), size=nb_samples)

    def lnprob(self, theta):

        if self.batchsize > 0:
            batch = [i % self.N0 for i in range(self.iter * self.batchsize, (self.iter + 1) * self.batchsize)]
            ridx = self.permutation[batch]
            self.iter += 1
        else:
            ridx = np.random.permutation(self.X.shape[0])

        Xs = self.X[ridx, :]
        Ys = self.Y[ridx]

        w = theta[:-1]  # logistic weights
        alpha = np.exp(theta[-1])  # the last column is logalpha
        d = len(w)
        coff = np.matmul(Xs, w.T)

        log_prior_alpha = gamma.logpdf(alpha, a=self.a0, scale=1/self.b0)
        log_prior_w = multivariate_normal.logpdf(w, mean=[0]*d, cov=1/alpha * np.identity(d))
        log_lik = np.log(1./(1. + np.exp(- Ys * coff)))

        return log_prior_alpha + log_prior_w + np.sum(log_lik)

    def evaluation(self, weights, theta, h_t, J_t, X_test, y_test, nb_samples):
        n_test = len(y_test)
        prob = np.zeros(n_test)
        dim_latent = len(theta[0])

        repartition = np.random.multinomial(nb_samples, weights)

        for j in range(J_t):
            nb = repartition[j]
            for _ in range(nb):
                y = self._generate_from_multivariate(theta[j], h_t, 1, dim_latent)[0]
                y = y[:-1]

                coff = np.multiply(y_test, np.sum(-1 * np.multiply(nm.repmat(y, n_test, 1), X_test), axis=1))
                prob += np.divide(np.ones(n_test), (1 + np.exp(coff)))

        prob = 1/nb_samples * prob
        predictive = np.mean(prob > 0.5)
        return predictive

# ### Main function ### #

def save_file(j, directory, strname, arraytosave):
    filename = directory + strname + str(j) + '.txt'
    np.savetxt(filename, arraytosave)


def main_function(j):
    print(j)
    # Save the results in different files

    directory = "./results/dim" + str(dim_latent) \
                + "/alpha" + str(alpha) \
                + "Jt" + str(J_t) \
                + "samples" + str(nb_samples_y_t) \
                + "T" + str(T) \
                + "N" + str(N_AGD) \
                + "/eta" + str(eta_n) \
                + "increment" + str(increment) + "/"
    Path(directory).mkdir(parents=True, exist_ok=True)
    func_eval = partial(model.evaluation, X_test=X_test, y_test=y_test, nb_samples=nb_eval)

    try:
        # Initialise (theta_1, ..., theta_J_t) and alpha0
        thetas_init = np.zeros([J_t, dim_latent])
        alpha0 = np.random.gamma(a0, b0, J_t)

        for i in range(J_t):
            thetas_init[i, :] = np.hstack([np.random.normal(0, np.sqrt(1 / alpha0[i]), d), np.log(alpha0[i])])

        powerDescent = AGD.AlphaGammaDescent(0, model.lnprob, dim_latent, thetas_init, alpha, T, N_AGD, J_t, h_t,
                                             nb_samples_y_t, eta_n, func_eval, False, False,
                                             increment, freq_eval)
        thetas, weights, renyi_bound_lst, llh_lst, evaluation_lst = powerDescent._full_algorithm()

        save_file(j, directory, 'PD_renyi_equi', renyi_bound_lst)
        save_file(j, directory, 'PD_llh_equi', llh_lst)
        save_file(j, directory, 'PD_pred_equi', evaluation_lst)

        if N_AGD > 1:
            powerDescent = AGD.AlphaGammaDescent(0, model.lnprob, dim_latent, thetas_init, alpha, T, N_AGD, J_t, h_t,
                                                 nb_samples_y_t, eta_n, func_eval, False, True,
                                                 increment, freq_eval)
            thetas, weights, renyi_bound_lst, llh_lst, evaluation_lst = powerDescent._full_algorithm()

            save_file(j, directory, 'PD_renyi_av_equi', renyi_bound_lst)
            save_file(j, directory, 'PD_llh_av_equi', llh_lst)
            save_file(j, directory, 'PD_pred_av_equi', evaluation_lst)

    except:
        pass

    return 0


# ### Load dataset ### #
data = scipy.io.loadmat('../../data/covertype.mat')

X_input = data['covtype'][:, 1:]
y_input = data['covtype'][:, 0]
y_input[y_input == 2] = -1

N = X_input.shape[0]
X_input = np.hstack([X_input, np.ones([N, 1])])
d = X_input.shape[1]
dim_latent = d + 1


# ### Split the dataset into training and testing ###
X_train, X_test, y_train, y_test = train_test_split(X_input, y_input, test_size=0.2, random_state=42)
print(len(y_train))

# ### Parameters ### #
J_t = 20
h_t = np.power(J_t, -1/(4 + dim_latent))
alpha = 0.5
T = 200
N_AGD = 1
increment = 1
freq_eval = 100
nb_eval = 100

nb_samples_y_t = J_t
eta_n = 0.05
eta_t = 1.

# ### Initialise Model ### #
a0, b0 = 1, 0.01  # hyper-parameters
model = BayesianLR(X_train, y_train, 100, a0, b0)  # batchsize = 100

# ### Launch the experiments ### #
nb_cores_used = 1#30
nb_repeat_exp = 1#0
i_list = range(nb_repeat_exp)
Parallel(nb_cores_used)(delayed(main_function)(i) for i in i_list)
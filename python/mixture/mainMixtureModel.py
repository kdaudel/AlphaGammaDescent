import numpy as np
from scipy.stats import multivariate_normal
from functools import partial
from joblib import Parallel, delayed
from pathlib import Path
import AlphaGammaDescent as AGD


# ### Model ### #

# Auxiliary functions
def _kernel_generate_1d(mean, sd, nb_samples):
    return np.random.normal(mean, sd, nb_samples)


def mixture_prob(y, means, sd, weights, nb_peaks, dim):
    '''
    Compute the pdf of the mixture model evaluated at y
    '''
    pdf_y = 0

    for i in range(nb_peaks):
        pdf_y += weights[i] * multivariate_normal.pdf(y, mean=means[i], cov=sd * np.identity(dim))

    return np.array(pdf_y)

class MVN:
    def __init__(self, target_nb_peaks, target_means, target_weights, target_sd, Z, D):
        '''
        Initialise a Mixture Model
        :param target_nb_peaks: number of modes (int)
        :param target_means: array of means (array)
        :param target_weights: mixture weights (array)
        :param target_sd: standard deviation (float > 0)
        :param Z: normalising constant (float > 0)
        :param D: dimension (int)
        '''
        self.target_nb_peaks = target_nb_peaks
        self.target_means = target_means
        self.target_weights = target_weights
        self.target_sd = target_sd
        self.Z = Z
        self.D = D

        if self.D == 1:
            self._generate_from_multivariate = _kernel_generate_1d
        else:
            self._generate_from_multivariate = self._kernel_generate_nd

    def lnprob(self, theta):
        '''
        Compute the log prob of the mixture model evaluated in theta
        '''
        prob = self.Z * mixture_prob(theta, self.target_means, self.target_sd, self.target_weights,
                                     self.target_nb_peaks, self.D)
        return np.log(prob)

    def _kernel_generate_nd(self, mean, sd, nb_samples):
        return np.random.multivariate_normal(mean=mean, cov=sd * np.identity(self.D), size=nb_samples)

    def sample_from_true_posterior(self, nb_samples_y):
        '''
        Sample according to the mixture model
        '''
        repartition = np.random.multinomial(nb_samples_y, self.target_weights)

        samples = []
        for i in range(self.target_nb_peaks):
            nb = repartition[i]
            u = self._generate_from_multivariate(self.target_means[i], self.target_sd, nb)
            samples.extend(u)
        return np.array(samples)


# ### Main function ### #

def save_file(j, directory, strname, arraytosave):
    '''
    Save array under the filename directory + strname + str(j) + '.txt'
    '''
    filename = directory + strname + str(j) + '.txt'
    np.savetxt(filename, arraytosave)


def main_function(j):

    # Initialise (theta_1, ..., theta_J_t) according to a normal distribution
    if dim_latent == 1:
        thetas_init = np.random.normal(q0_mean, q0_sd, J_t)
    else:
        thetas_init = np.random.multivariate_normal(q0_mean, q0_sd * np.identity(dim_latent), J_t)

    try:
        if not alpha == 1.:
            powerDescent = AGD.AlphaGammaDescent(0, model.lnprob, dim_latent, thetas_init, alpha, T, N, J_t, h_t,
                                                  nb_samples_y_t, eta_n, False, False)
            thetas_final, weights_final, renyi_bound_lst, llh_lst = powerDescent._full_algorithm()

            save_file(j, directory, 'mixture_PD_renyi_not_av', renyi_bound_lst)
            save_file(j, directory, 'mixture_PD_llh_not_av', llh_lst)

            # Alternative version of the Power descent with lambda_N = 1/N sum_n lambda_n
            powerDescentAv = AGD.AlphaGammaDescent(0, model.lnprob, dim_latent, thetas_init, alpha, T, N, J_t, h_t,
                                                     nb_samples_y_t, eta_n, False, True)
            thetas_final, weights_final, renyi_bound_lst, llh_lst = powerDescentAv._full_algorithm()

            save_file(j, directory, 'mixture_PD_renyi_av', renyi_bound_lst)
            save_file(j, directory, 'mixture_PD_llh_av', llh_lst)

        mirrorDescent = AGD.AlphaGammaDescent(0, model.lnprob, dim_latent, thetas_init, alpha, T, N, J_t, h_t,
                                              nb_samples_y_t, eta_n, True, False)
        thetas_final_md, weights_final_md, renyi_bound_lst_md, llh_lst_md = mirrorDescent._full_algorithm()

        save_file(j, directory, 'mixture_MD_renyi_md_not_av', renyi_bound_lst_md)
        save_file(j, directory, 'mixture_MD_llh_md_not_av', llh_lst_md)

        # Alternative version of the Entropic Mirror descent with lambda_N = 1/N sum_n lambda_n
        mirrorDescentAv = AGD.AlphaGammaDescent(0, model.lnprob, dim_latent, thetas_init, alpha, T, N, J_t, h_t,
                                                nb_samples_y_t, eta_n, True, True)
        thetas_final_md, weights_final_md, renyi_bound_lst_md, llh_lst_md = mirrorDescentAv._full_algorithm()

        save_file(j, directory, 'mixture_MD_renyi_md_av', renyi_bound_lst_md)
        save_file(j, directory, 'mixture_MD_llh_md_av', llh_lst_md)

    except:
        pass

    return 0


# ### Parameters ### #
alpha = 0.5
T = 10
N = 10
J_t = 100
nb_samples_y_t = 100
dim_latent = 8
eta_n = 0.5
h_t = np.power(J_t, -1/(4 + dim_latent))

q0_sd = 5
q0_mean = np.array([0] * dim_latent)

directory = './results/dim' + str(dim_latent) + "/alpha" + str(alpha) + "/eta" + str(eta_n) + '/'
Path(directory).mkdir(parents=True, exist_ok=True)

# ### Define the targeted density and initialise model ### #
target_means = [[2] * dim_latent, [-2] * dim_latent]
target_nb_peaks = len(target_means)
target_sd = 1.
target_weights = [1 / target_nb_peaks] * target_nb_peaks
Z = 2

model = MVN(target_nb_peaks, target_means, target_weights, target_sd, Z, dim_latent)

# ### Launch the experiments ### #
nb_cores_used = 1#50
nb_repeat_exp = 10#00
i_list = range(nb_repeat_exp)
Parallel(nb_cores_used)(delayed(main_function)(i) for i in i_list)
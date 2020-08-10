import matplotlib.pyplot as plt
import numpy as np
import glob
import re


def read_res(filename):
    f1 = open(filename, 'r')
    fileline = f1.read()
    fileline = re.split(' |\n', fileline)
    return np.array([float(fileline[j]) for j in range(len(fileline)-1)])


def wrapper_function(directory, T, strName2):
    f_renyi = glob.glob(directory + strName2)
    nb_repeat = len(f_renyi)
    summation = np.zeros(T)

    for i in range(nb_repeat):
        file_read_llh = read_res(f_renyi[i])
        summation = summation + np.array(file_read_llh)

    summation = summation / nb_repeat

    return summation


alpha = 0.5
T = 10
N = 10
J_t = 100
nb_samples_y_t = 100
dim_latent = 8
eta_n = 0.5
h_t = np.power(J_t, -1/(4 + dim_latent))
directory = './results/dim' + str(dim_latent) + "/alpha" + str(alpha) + "/eta" + str(eta_n) + '/'

plt.style.use('bmh')
plt.rcParams['figure.facecolor'] = '1'
plt.rcParams['axes.facecolor'] = 'w'
plt.rcParams["legend.loc"] = 'lower right'#'upper left'

#"""
if not alpha == 1:
    summation_PD_renyi_av = wrapper_function(directory, T * N, 'mixture_PD_renyi_av*.txt')

    summation_PD_renyi_not_av = wrapper_function(directory, T * N, 'mixture_PD_renyi_not_av*.txt')

summation_MD_renyi_av = wrapper_function(directory, T * N, 'mixture_MD_renyi_md_av*.txt')

summation_MD_renyi_not_av = wrapper_function(directory, T * N, 'mixture_MD_renyi_md_not_av*.txt')

plt.figure()
plt.title('Dimension ' + str(dim_latent) + r', $\alpha$ = ' + str(alpha) + r', $\eta_0$ = ' + str(eta_n))
plt.axhline(y=np.log(2), c='grey', label=r'$\log Z$')
if not alpha == 1.:
    plt.plot(summation_PD_renyi_not_av, label=str(alpha)+'-Power')
    plt.plot(summation_PD_renyi_av, label=str(alpha) +'-Power avg')

    plt.plot(summation_MD_renyi_not_av, label=str(alpha)+'-Mirror')
    plt.plot(summation_MD_renyi_av, label=str(alpha)+'-Mirror avg')

if alpha == 1.:
    plt.plot(summation_MD_renyi_not_av, label='1.-Mirror')
    plt.plot(summation_MD_renyi_av, label='1.-Mirror avg')

plt.xlabel('Iterations')
plt.ylabel('Renyi Bound')
plt.legend(loc='lower right')
plt.show()
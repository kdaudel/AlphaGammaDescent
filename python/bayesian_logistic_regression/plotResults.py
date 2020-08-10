import matplotlib.pyplot as plt
import numpy as np
import os
import glob

def read_res(filename):
    array = []
    f1 = open(filename, 'r')
    u = f1.readline()
    try:
        array.append(float(u))
    except:
        #print("missing val")
        pass

    while len(u) != 0:
        u = f1.readline()
        try:
            array.append(float(u))
        except:
            pass
    f1.close()
    return array


def wrapper_function(directory, T, N_fei, strName1, strName2):

    f_pred = glob.glob(directory + strName1)
    f = glob.glob(directory + strName2)

    nb_repeat = len(f_pred)
    print("total", nb_repeat)
    summation_pred = np.zeros(eval_nb)
    summation = np.zeros(T*N_fei)
    count = 0

    for i in range(nb_repeat):
        file_read_acc = read_res(f_pred[i])
        file_read = read_res(f[i])

        val = np.array(file_read)
        is_infinite_value = - np.inf in val

        if not is_infinite_value:
            summation_pred = summation_pred + np.array(file_read_acc)
            summation = summation + np.array(file_read)
        else :
            count +=1
            nb_repeat -=1

    summation_pred = summation_pred / nb_repeat
    summation = summation / nb_repeat

    print("inf_value", count)
    print("total_left", nb_repeat)
    return summation_pred, summation


def wrapper_function_llh(directory, T, strName1):

    f_llh = glob.glob(directory + strName1)

    nb_repeat = len(f_llh)
    summation_llh = np.zeros(T)
    print("total", nb_repeat)

    count = 0
    for i in range(nb_repeat):
        file_read_llh = read_res(f_llh[i])

        val = np.array(file_read_llh)
        is_infinite_value = - np.inf in val

        if not is_infinite_value:
            summation_llh = summation_llh + np.array(file_read_llh)

        else :
            count +=1
            nb_repeat -=1

    summation_llh = summation_llh / nb_repeat
    print("inf_value", count)
    print("total_left", nb_repeat)

    return summation_llh

dim = 56
J_t = 20
h_t = np.power(J_t, -1/(4 + dim))
alpha = 0.5
T = 200
N_AGD = 1
increment = 1
freq_eval = 100
nb_eval = 100

nb_samples_y_t = J_t
eta_n = 0.05
eta_t = 1.

eval_nb = int(T/freq_eval+1) # 501

plt.style.use('bmh')
plt.rcParams['figure.facecolor'] = '1'
plt.rcParams['axes.facecolor'] = 'w'
plt.rcParams["legend.loc"] = 'lower right'#'upper left'

N_val = [1]
nb_samples_y_t_val = [J_t]
for i in N_val:
    for k in nb_samples_y_t_val:
        directory = "./results/dim" + str(dim) \
                    + "/alpha" + str(alpha) \
                    + "Jt" + str(J_t) \
                    +"samples" + str(k) \
                    + "T" + str(T) \
                    + "N" + str(i) \
                    + "/eta" + str(eta_n) \
                    + "increment" + str(increment) + "/"
        k = int(k)

        print("aei")
        summation_aei_pred, summation_aei_renyi = wrapper_function(directory, T, i, 'PD_pred_equi*.txt',
                                                                   'PD_renyi_equi*.txt')
        summation_aei_llh = wrapper_function_llh(directory, T, 'PD_llh_equi*.txt')

        plt.figure()
        plt.plot(summation_aei_llh, label=r'$0.5$-Power')
        plt.xlabel('Iterations')
        plt.ylabel('Log-likelihood')
        plt.legend()

        plt.figure()
        plt.plot(summation_aei_renyi, label=r'$0.5$-Power')
        plt.xlabel('Iterations')
        plt.ylabel('Renyi Bound')
        plt.legend()

        plt.figure()
        u = np.array(range(eval_nb)) * freq_eval
        plt.plot(u,summation_aei_pred, label=r'$0.5$-Power')
        plt.xlabel('Iterations')
        plt.ylabel('Accuracy')
        plt.legend()

plt.show()
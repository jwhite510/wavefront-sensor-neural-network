import argparse
import matplotlib.pyplot as plt
import numpy as np
import pickle

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--pc',nargs='+')
    args=parser.parse_args()
    peakcounts=[]
    errorvals=[]
    for pc in args.pc:
        with open('error_'+pc+'.p','rb') as file:
            obj=pickle.load(file)
            peakcounts.append(pc)
            errorvals.append(obj)

    average=[]
    stdev=[]

    for _err in errorvals:
        _err['network_error_phase_avg'] = np.average(_err['network_error_phase'])
        _err['network_error_phase_std'] = np.std(_err['network_error_phase'])
        _err['network_error_intensity_avg'] = np.average(_err['network_error_intensity'])
        _err['network_error_intensity_std'] = np.std(_err['network_error_intensity'])
        _err['iterative_error_phase_avg'] = np.average(_err['iterative_error_phase'])
        _err['iterative_error_phase_std'] = np.std(_err['iterative_error_phase'])
        _err['iterative_error_intensity_avg'] = np.average(_err['iterative_error_intensity'])
        _err['iterative_error_intensity_std'] = np.std(_err['iterative_error_intensity'])

    # plot intensity average + std
    network_avg=[]
    network_std=[]
    iterative_avg=[]
    iterative_std=[]
    for _err in errorvals:
        network_avg.append(_err['network_error_intensity_avg'])
        network_std.append(_err['network_error_intensity_std'])
        iterative_avg.append(_err['iterative_error_intensity_avg'])
        iterative_std.append(_err['iterative_error_intensity_std'])

    plt.figure(1)
    plt.errorbar(np.array(peakcounts,dtype=int),network_avg,network_std,label='network_avg',alpha=0.5)
    plt.errorbar(np.array(peakcounts,dtype=int),iterative_avg,iterative_std,label='iterative_avg',alpha=0.5)
    plt.legend()
    plt.show()

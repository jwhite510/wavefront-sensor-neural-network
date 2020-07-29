import argparse
import matplotlib.pyplot as plt
import numpy as np
import pickle
from CompareNN_MatlabBilinearInterp import PhaseIntensityError, ErrorDistribution




def construct_arrays(errorvals,error_type):
    """
    errorvals : list of PhaseIntensityError objects
    error_type : string -> 'phase' or 'intensity'
    """
    network_avg=[]
    network_std=[]
    iterative_avg=[]
    iterative_std=[]
    for _err in errorvals:
        if error_type == 'phase':
            network_avg.append(_err['network_error'].phase_error.average)
            network_std.append(_err['network_error'].phase_error.standard_deviation)
            iterative_avg.append(_err['iterative_error'].phase_error.average)
            iterative_std.append(_err['iterative_error'].phase_error.standard_deviation)
        elif error_type == 'intensity':
            network_avg.append(_err['network_error'].intensity_error.average)
            network_std.append(_err['network_error'].intensity_error.standard_deviation)
            iterative_avg.append(_err['iterative_error'].intensity_error.average)
            iterative_std.append(_err['iterative_error'].intensity_error.standard_deviation)

    return network_avg,network_std,iterative_avg,iterative_std


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

    # construct list of average and standard deviation values
    # plot intensity average + std
    network_avg,network_std,iterative_avg,iterative_std=construct_arrays(errorvals,'intensity')
    plt.figure(1)
    plt.title("intensity")
    plt.errorbar(np.array(peakcounts,dtype=int),network_avg,network_std,label='neural network',alpha=0.5)
    plt.errorbar(np.array(peakcounts,dtype=int),iterative_avg,iterative_std,label='iterative',alpha=0.5)
    plt.gca().set_xlabel("peak signal count")
    plt.gca().set_ylabel("Mean Square Error (intensity)")
    plt.legend()
    plt.savefig('intensity.png')

    # plot intensity average + std
    network_avg,network_std,iterative_avg,iterative_std=construct_arrays(errorvals,'phase')

    plt.figure(2)
    plt.title("phase")
    plt.errorbar(np.array(peakcounts,dtype=int),network_avg,network_std,label='neural network',alpha=0.5)
    plt.errorbar(np.array(peakcounts,dtype=int),iterative_avg,iterative_std,label='iterative',alpha=0.5)
    plt.gca().set_xlabel("peak signal count")
    plt.gca().set_ylabel("Mean Square Error (phase)")
    plt.legend()
    plt.savefig('phase.png')

    plt.show()

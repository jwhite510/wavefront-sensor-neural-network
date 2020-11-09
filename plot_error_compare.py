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
    args,_=parser.parse_known_args()
    peakcounts=[]
    errorvals=[]
    for pc in args.pc:
        with open('error_'+pc+'.p','rb') as file:
            obj=pickle.load(file)
            peakcounts.append(pc)
            errorvals.append(obj)

    # construct list of average and standard deviation values
    # plot intensity average + std
    fig=plt.figure()
    fig.subplots_adjust(left=0.15,hspace=0.0)
    gs=fig.add_gridspec(2,1)
    letter='a'
    for _i,_retrieval in enumerate(['intensity','phase']):
        ax=fig.add_subplot(gs[_i,0])
        network_avg,network_std,iterative_avg,iterative_std=construct_arrays(errorvals,_retrieval)
        ax.text(0.03,0.9,letter,transform=ax.transAxes,weight='bold',backgroundcolor='white');letter=chr(ord(letter)+1)
        ax.text(0.1,0.9,"Retrieved "+('Intensity'if _retrieval=='intensity'else'Phase')+" Error",transform=ax.transAxes)
        ax.errorbar(np.array(peakcounts,dtype=int),network_avg,network_std,label='neural network',alpha=0.5,color='blue',linewidth=4.0)
        ax.errorbar(np.array(peakcounts,dtype=int),iterative_avg,iterative_std,label='iterative',alpha=0.5,color='orange',linewidth=4.0)
        if _i==0:ax.set_xticks([]);ax.legend(bbox_to_anchor=(0.65,1.3),loc='upper left');ax.set_ylim([0.01,0.14])
        else: ax.set_xlabel("peak signal count");ax.set_ylim([0.01,0.7])
        ax.set_ylabel("Average Root Mean\nSquare Error ("+('Intensity'if _retrieval=='intensity'else'Phase')+")")
    fig.savefig('intensity_phase_nn_it_compared.png')
    # plt.show()

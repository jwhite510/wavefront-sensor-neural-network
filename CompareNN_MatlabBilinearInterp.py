import numpy as np
import scipy
import diffraction_functions
import matplotlib.pyplot as plt
import diffraction_net
import tables
import pickle
import os
from scipy import interpolate
import argparse

def get_interpolation_points(amplitude_mask):
    """
        get the points for bilinear interp
    """
    x=[]
    y=[]

    # plt.figure()
    # plt.title("Left Points")
    # plt.pcolormesh(amplitude_mask)
    for col in [35,41,48,54,61]:
        for row in [93,86,80,74,67,61,54,48,41,35]:
            x.append(col)
            y.append(row)
            # plt.axvline(x=col,color="red")
            # plt.axhline(y=row,color="blue")

    # plt.figure()
    # plt.title("Right Upper Points")
    # plt.pcolormesh(amplitude_mask)
    for col in [67,73,80,86,93]:
        for row in [93,86,80,74,67]:
            x.append(col)
            y.append(row)
            # plt.axvline(x=col,color="red")
            # plt.axhline(y=row,color="blue")

    # plt.figure()
    # plt.title("Right Lower Points")
    # plt.pcolormesh(amplitude_mask)
    for col in [66,72,79,85,91]:
        for row in [62,55,49,42,36]:
            x.append(col)
            y.append(row)
            # plt.axvline(x=col,color="red")
            # plt.axhline(y=row,color="blue")

    return x,y


if __name__ == "__main__":

    parser=argparse.ArgumentParser()
    parser.add_argument('--network',type=str)
    parser.add_argument('--IMAGE_ANNOTATE',type=str)
    parser.add_argument('--SAVE_FOLDER',type=str)
    args=parser.parse_args()
    IMAGE_ANNOTATE=args.IMAGE_ANNOTATE
    SAVE_FOLDER=args.SAVE_FOLDER
    # IMAGE_ANNOTATE="_c_INF"
    # SAVE_FOLDER="test_7_6_c50"
    # set the center for plotting the phase to compare all the figures
    m_index=(64,64)
    # load diffraction pattern
    os.mkdir(SAVE_FOLDER)
    index=11
    # index=9 # best
    N=None
    with tables.open_file("zernike3/build/test_noise.hdf5",mode="r") as file:
    # with tables.open_file("zernike3/build/test.hdf5",mode="r") as file: # use the noise free sample, and matlab result looks good
        N = file.root.N[0,0]
        object_real = file.root.object_real[index, :].reshape(N,N)
        object_imag = file.root.object_imag[index, :].reshape(N,N)
        diffraction = file.root.diffraction[index, :].reshape(N,N)
    # no noise diffraction pattern
    with tables.open_file("zernike3/build/test.hdf5",mode="r") as file:
        N = file.root.N[0,0]
        diffraction_noisefree = file.root.diffraction[index, :].reshape(N,N)

    actual_object = {}
    actual_object["measured_pattern"] = diffraction
    actual_object["tf_reconstructed_diff"] = diffraction_noisefree
    actual_object["real_output"] = object_real
    actual_object["imag_output"] = object_imag

    # get the peak of the intensity pattern
    fig=diffraction_functions.plot_amplitude_phase_meas_retreival(actual_object,"actual_object_MASKED"+IMAGE_ANNOTATE,ACTUAL=True,m_index=m_index,mask=True)
    fig.savefig(os.path.join(SAVE_FOLDER,"actual_object_MASKED"+IMAGE_ANNOTATE+".png"))

    fig=diffraction_functions.plot_amplitude_phase_meas_retreival(actual_object,"actual_object"+IMAGE_ANNOTATE,ACTUAL=True,m_index=m_index)
    fig.savefig(os.path.join(SAVE_FOLDER,"actual_object"+IMAGE_ANNOTATE+".png"))

    # retrieve image with neural network
    network=diffraction_net.DiffractionNet(args.network) # load a pre trained network

    # get the reconstructed diffraction pattern and the real / imaginary object
    nn_retrieved = {}

    nn_retrieved["measured_pattern"] = diffraction

    nn_retrieved["tf_reconstructed_diff"] = network.sess.run(
            network.nn_nodes["recons_diffraction_pattern"], feed_dict={network.x:diffraction.reshape(1,N,N,1)})

    nn_retrieved["real_output"] = network.sess.run(
            network.nn_nodes["real_out"], feed_dict={network.x:diffraction.reshape(1,N,N,1)})

    nn_retrieved["imag_output"] = network.sess.run(
            network.nn_nodes["imag_out"], feed_dict={network.x:diffraction.reshape(1,N,N,1)})

    # plot retrieval with neural network
    fig=diffraction_functions.plot_amplitude_phase_meas_retreival(nn_retrieved,"nn_retrieved"+IMAGE_ANNOTATE,m_index=m_index)
    fig.savefig(os.path.join(SAVE_FOLDER,"nn_retrieved"+IMAGE_ANNOTATE+".png"))

    # get amplitude mask
    N = np.shape(nn_retrieved["measured_pattern"])[1]
    _, amplitude_mask = diffraction_functions.get_amplitude_mask_and_imagesize(N, int(N/2))
    # get interpolation points

    # run matlab retrieval with and without interpolation
    matlabcdi_retrieved_interp=diffraction_functions.matlab_cdi_retrieval(np.squeeze(nn_retrieved['measured_pattern']),amplitude_mask,interpolate=True)
    fig=diffraction_functions.plot_amplitude_phase_meas_retreival(matlabcdi_retrieved_interp,"matlabcdi_retrieved_interp"+IMAGE_ANNOTATE,m_index=m_index)
    fig.savefig(os.path.join(SAVE_FOLDER,"matlabcdi_retrieved_interp"+IMAGE_ANNOTATE+".png"))

    # noise reduction
    matlabcdi_retrieved_interp_NR=diffraction_functions.matlab_cdi_retrieval(np.squeeze(nn_retrieved['measured_pattern']),amplitude_mask,interpolate=True,noise_reduction=True)
    fig=diffraction_functions.plot_amplitude_phase_meas_retreival(matlabcdi_retrieved_interp_NR,"matlabcdi_retrieved_interp_NR"+IMAGE_ANNOTATE,m_index=m_index)
    fig.savefig(os.path.join(SAVE_FOLDER,"matlabcdi_retrieved_interp_NR"+IMAGE_ANNOTATE+".png"))


    # matlabcdi_retrieved_NOinterp=diffraction_functions.matlab_cdi_retrieval(np.squeeze(nn_retrieved['measured_pattern']),amplitude_mask,interpolate=False)
    # fig=diffraction_functions.plot_amplitude_phase_meas_retreival(matlabcdi_retrieved_NOinterp,"matlabcdi_retrieved_NOinterp"+IMAGE_ANNOTATE,m_index=m_index)
    # fig.savefig(os.path.join(SAVE_FOLDER,"matlabcdi_retrieved_NOinterp"+IMAGE_ANNOTATE+".png"))

    # matlabcdi_retrieved_NOinterp_NR=diffraction_functions.matlab_cdi_retrieval(np.squeeze(nn_retrieved['measured_pattern']),amplitude_mask,interpolate=False,noise_reduction=True)
    # fig=diffraction_functions.plot_amplitude_phase_meas_retreival(matlabcdi_retrieved_NOinterp_NR,"matlabcdi_retrieved_NOinterp_NR"+IMAGE_ANNOTATE,m_index=m_index)
    # fig.savefig(os.path.join(SAVE_FOLDER,"matlabcdi_retrieved_NOinterp_NR"+IMAGE_ANNOTATE+".png"))

    # plt.show()







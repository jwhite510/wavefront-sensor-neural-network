import numpy as np
from numpy import unravel_index
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

class CompareNetworkIterative():
    def __init__(self, args):
        # retrieve image with neural network
        self.network=diffraction_net.DiffractionNet(args.network) # load a pre trained network
        self.args=args

    def test(self):
        m_index=(64,64)
        # load diffraction pattern
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

        fig=diffraction_functions.plot_amplitude_phase_meas_retreival(actual_object,"actual_object",ACTUAL=True,m_index=m_index,mask=False)

        # get the reconstructed diffraction pattern and the real / imaginary object
        nn_retrieved = {}
        nn_retrieved["measured_pattern"] = diffraction
        nn_retrieved["tf_reconstructed_diff"] = self.network.sess.run(
                self.network.nn_nodes["recons_diffraction_pattern"], feed_dict={self.network.x:diffraction.reshape(1,N,N,1)})
        nn_retrieved["real_output"] = self.network.sess.run(
                self.network.nn_nodes["real_out"], feed_dict={self.network.x:diffraction.reshape(1,N,N,1)})
        nn_retrieved["imag_output"] = self.network.sess.run(
                self.network.nn_nodes["imag_out"], feed_dict={self.network.x:diffraction.reshape(1,N,N,1)})

        # plot retrieval with neural network
        # fig=diffraction_functions.plot_amplitude_phase_meas_retreival(nn_retrieved,"nn_retrieved",m_index=m_index)

        # get amplitude mask
        N = np.shape(nn_retrieved["measured_pattern"])[1]
        _, amplitude_mask = diffraction_functions.get_amplitude_mask_and_imagesize(N, int(N/2))
        # get interpolation points

        # run matlab retrieval with and without interpolation
        # matlabcdi_retrieved_interp=diffraction_functions.matlab_cdi_retrieval(np.squeeze(nn_retrieved['measured_pattern']),amplitude_mask,interpolate=True)
        # with open("matlab_cdi_retrieval.p","wb") as file:
            # pickle.dump(matlabcdi_retrieved_interp,file)
        with open("matlab_cdi_retrieval.p","rb") as file:
            matlabcdi_retrieved_interp=pickle.load(file)

        fig=diffraction_functions.plot_amplitude_phase_meas_retreival(matlabcdi_retrieved_interp,"matlabcdi_retrieved_interp",m_index=m_index)

        # compare and calculate phase + intensity error
        # plt.close('all')
        phase_mse,intensity_mse=intensity_phase_error(actual_object,matlabcdi_retrieved_interp)
        # rmse=intensity_phase_error(actual_object,nn_retrieved)
        # actual_object
        # matlabcdi_retrieved_interp
        # nn_retrieved
        plt.show()

def intensity_phase_error(actual,predicted):
    """
    actual, predicted
    : dictionaries with keys:

    measured_pattern
    tf_reconstructed_diff
    real_output
    imag_output

    """
    actual_c = actual["real_output"]+1j*actual["imag_output"]
    predicted_c = predicted["real_output"]+1j*predicted["imag_output"]

    # set both to 0 at less than 50% predicted peak
    actual_c[np.abs(predicted_c)**2 < 0.05 * np.max(np.abs(predicted_c)**2)] = 0.0
    predicted_c[np.abs(predicted_c)**2 < 0.05 * np.max(np.abs(predicted_c)**2)] = 0.0

    actual_I = np.abs(actual_c)**2
    predicted_I = np.abs(predicted_c)**2

    # find intensity peak of predicted
    m_index = unravel_index(predicted_I.argmax(), predicted_I.shape)
    predicted_phase_Imax = np.angle(predicted_c[m_index[0], m_index[1]])
    actual_phase_Imax = np.angle(actual_c[m_index[0], m_index[1]])
    # subtract phase at center
    predicted_c *= np.exp(-1j * predicted_phase_Imax)
    actual_c *= np.exp(-1j * actual_phase_Imax)

    # phase rmse
    A = np.angle(actual_c).reshape(-1)
    B = np.angle(predicted_c).reshape(-1)
    phase_mse = (np.square(A-B)).mean()

    A = actual_I.reshape(-1)
    B = predicted_I.reshape(-1)
    intensity_mse = (np.square(A-B)).mean()


    plt.figure(101)
    plt.title("actual_c")
    plt.imshow(np.angle(actual_c))
    plt.gca().axvline(x=m_index[1],color="red",alpha=0.8)
    plt.gca().axhline(y=m_index[0],color="blue",alpha=0.8)
    plt.colorbar()

    plt.figure(102)
    plt.title("actual_c")
    plt.plot(np.angle(actual_c)[m_index[0],:])
    plt.gca().axvline(x=m_index[1],color="red")

    plt.figure(103)
    plt.title("predicted_c")
    plt.imshow(np.angle(predicted_c))
    plt.gca().axvline(x=m_index[1],color="red",alpha=0.8)
    plt.gca().axhline(y=m_index[0],color="blue",alpha=0.8)
    plt.colorbar()

    plt.figure(104)
    plt.title("predicted_c")
    plt.plot(np.angle(predicted_c)[m_index[0],:])
    plt.gca().axvline(x=m_index[1],color="red")

    # plt.figure(105)
    # plt.imshow(np.angle(predicted_c) - np.angle(actual_c))
    # plt.gca().axvline(x=m_index[1],color="red",alpha=0.8)
    # plt.gca().axhline(y=m_index[0],color="blue",alpha=0.8)
    # plt.colorbar()

    return phase_mse,intensity_mse




if __name__ == "__main__":

    # TODO : evaluate rmse at high intensity areas
    # + phase, set constant phase shift

    # evaluate at different noise levels

    # run a variational network, run an RNN network

    parser=argparse.ArgumentParser()
    parser.add_argument('--network',type=str)
    parser.add_argument('--IMAGE_ANNOTATE',type=str)
    parser.add_argument('--SAVE_FOLDER',type=str)
    args=parser.parse_args()
    comparenetworkiterative = CompareNetworkIterative(args)
    comparenetworkiterative.test()



import numpy as np
import scipy
import diffraction_functions
import matplotlib.pyplot as plt
import diffraction_net
import tables
import pickle
import os
from scipy import interpolate

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

    # # retrieve neural network result
    # folder_dir="nn_pictures/teslatest5_doubleksize_doublefilters_reconscostfunction_pictures/46/measured/"
    # # run_name="Data_for_Jonathon_z0_1-fits_ud_1-0_reconstructed.p"
    # sample_name="Data_for_Jonathon_z-500_1-fits_ud_1-0_reconstructed.p"
    # filename=os.path.join(folder_dir,sample_name)
    # print("filename =>", filename)
    # with open(filename,"rb") as file:
        # nn_retrieved=pickle.load(file)

    # load diffraction pattern
    index=10
    N=None
    with tables.open_file("zernike3/build/test_noise.hdf5",mode="r") as file:
        N = file.root.N[0,0]
        object_real = file.root.object_real[index, :].reshape(N,N)
        object_imag = file.root.object_imag[index, :].reshape(N,N)
        diffraction = file.root.diffraction[index, :].reshape(N,N)

    plt.figure()
    plt.title("object_real actual")
    plt.imshow(object_real)

    plt.figure()
    plt.title("object_imag actual")
    plt.imshow(object_imag)

    plt.figure()
    plt.title("diffraction")
    plt.imshow(diffraction)


    # retrieve image with neural network
    network=DiffractionNet("networkname") # load a pre trained network

    # get the reconstructed diffraction pattern and the real / imaginary object
    nn_retrieved = {}

    nn_retrieved["measured_pattern"] = diffraction

    nn_retrieved["tf_reconstructed_diff"] = network.sess.run(
            network.nn_nodes["recons_diffraction_pattern"], feed_dict={network.x:diffraction})

    nn_retrieved["real_output"] = network.sess.run(
            network.nn_nodes["real_out"], feed_dict={network.x:diffraction})

    nn_retrieved["imag_output"] = network.sess.run(
            network.nn_nodes["imag_out"], feed_dict={network.x:diffraction})

    plt.show()
    exit()

    # plot retrieval with neural network
    diffraction_functions.plot_amplitude_phase_meas_retreival(nn_retrieved,"nn_retrieved")

    # get amplitude mask
    N = np.shape(nn_retrieved["measured_pattern"])[1]
    _, amplitude_mask = diffraction_functions.get_amplitude_mask_and_imagesize(N, int(N/2))
    # get interpolation points

    # run matlab retrieval with and without interpolation
    matlabcdi_retrieved_interp=diffraction_functions.matlab_cdi_retrieval(np.squeeze(nn_retrieved['measured_pattern']),amplitude_mask,interpolate=True)
    matlabcdi_retrieved_NOinterp=diffraction_functions.matlab_cdi_retrieval(np.squeeze(nn_retrieved['measured_pattern']),amplitude_mask,interpolate=False)

    diffraction_functions.plot_amplitude_phase_meas_retreival(matlabcdi_retrieved_interp,"matlabcdi_retrieved_interp")
    diffraction_functions.plot_amplitude_phase_meas_retreival(matlabcdi_retrieved_NOinterp,"matlabcdi_retrieved_NOinterp")

    plt.show()







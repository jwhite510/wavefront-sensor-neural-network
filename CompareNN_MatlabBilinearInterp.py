import numpy as np
import scipy
import diffraction_functions
import matplotlib.pyplot as plt
import pickle
import os

if __name__ == "__main__":

    # retrieve neural network result
    folder_dir="nn_pictures/teslatest5_doubleksize_doublefilters_reconscostfunction_pictures/46/measured/"
    # run_name="Data_for_Jonathon_z0_1-fits_ud_1-0_reconstructed.p"
    sample_name="Data_for_Jonathon_z-500_1-fits_ud_1-0_reconstructed.p"
    filename=os.path.join(folder_dir,sample_name)
    print("filename =>", filename)
    with open(filename,"rb") as file:
        nn_retrieved=pickle.load(file)
    # complex_beam=np.squeeze(nn_retrieved["real_output"]+1j*nn_retrieved["imag_output"])

    diffraction_functions.plot_amplitude_phase_meas_retreival(nn_retrieved,"nn_retrieved")

    # run retreival on this with matlab
    print(nn_retrieved.keys())
    plt.figure()
    plt.pcolormesh(np.squeeze(nn_retrieved['measured_pattern']))
    plt.title("measured_pattern")

    # get amplitude mask
    N = np.shape(nn_retrieved["measured_pattern"])[1]
    _, amplitude_mask = diffraction_functions.get_amplitude_mask_and_imagesize(N, int(N/2))
    plt.figure()
    plt.pcolormesh(amplitude_mask)
    plt.title("amplitude_mask")

    # retrieve the object from matlab CDI code
    matlabcdi_retrieved=diffraction_functions.matlab_cdi_retrieval(np.squeeze(nn_retrieved['measured_pattern']),amplitude_mask)
    diffraction_functions.plot_amplitude_phase_meas_retreival(matlabcdi_retrieved,"matlabcdi_retrieved")


    plt.show()






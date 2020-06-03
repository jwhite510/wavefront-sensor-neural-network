import numpy as np
import scipy
import diffraction_functions
import matplotlib.pyplot as plt
import pickle
import os

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

    # retrieve neural network result
    folder_dir="nn_pictures/teslatest5_doubleksize_doublefilters_reconscostfunction_pictures/46/measured/"
    # run_name="Data_for_Jonathon_z0_1-fits_ud_1-0_reconstructed.p"
    sample_name="Data_for_Jonathon_z-500_1-fits_ud_1-0_reconstructed.p"
    filename=os.path.join(folder_dir,sample_name)
    print("filename =>", filename)
    with open(filename,"rb") as file:
        nn_retrieved=pickle.load(file)
    # complex_beam=np.squeeze(nn_retrieved["real_output"]+1j*nn_retrieved["imag_output"])

    # diffraction_functions.plot_amplitude_phase_meas_retreival(nn_retrieved,"nn_retrieved")

    # run retreival on this with matlab
    print(nn_retrieved.keys())
    # plt.figure()
    # plt.pcolormesh(np.squeeze(nn_retrieved['measured_pattern']))
    # plt.title("measured_pattern")

    # get amplitude mask
    N = np.shape(nn_retrieved["measured_pattern"])[1]
    _, amplitude_mask = diffraction_functions.get_amplitude_mask_and_imagesize(N, int(N/2))
    # get interpolation points
    points=get_interpolation_points(amplitude_mask)

    plt.figure()
    plt.pcolormesh(amplitude_mask)
    plt.title("amplitude_mask")

    # retrieve the object from matlab CDI code
    matlabcdi_retrieved=diffraction_functions.matlab_cdi_retrieval(np.squeeze(nn_retrieved['measured_pattern']),amplitude_mask)
    diffraction_functions.plot_amplitude_phase_meas_retreival(matlabcdi_retrieved,"matlabcdi_retrieved")




    # check if t
    matlab_complex_object=matlabcdi_retrieved["real_output"]+1j*matlabcdi_retrieved["imag_output"]
    plt.figure(10)
    plt.pcolormesh(np.real(matlab_complex_object))
    plt.figure(11)
    plt.pcolormesh(np.imag(matlab_complex_object))



    plt.show()






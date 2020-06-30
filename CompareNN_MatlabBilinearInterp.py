import numpy as np
import scipy
import diffraction_functions
import matplotlib.pyplot as plt
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
    # plt.figure()
    # plt.pcolormesh(np.squeeze(nn_retrieved['measured_pattern']))
    # plt.title("measured_pattern")

    # get amplitude mask
    N = np.shape(nn_retrieved["measured_pattern"])[1]
    _, amplitude_mask = diffraction_functions.get_amplitude_mask_and_imagesize(N, int(N/2))
    # get interpolation points
    x,y=points=get_interpolation_points(amplitude_mask)

    plt.figure()
    plt.pcolormesh(amplitude_mask)
    plt.title("amplitude_mask")

    # retrieve the object from matlab CDI code
    matlabcdi_retrieved_interp=diffraction_functions.matlab_cdi_retrieval(np.squeeze(nn_retrieved['measured_pattern']),amplitude_mask,interpolate=True)
    matlabcdi_retrieved_NOinterp=diffraction_functions.matlab_cdi_retrieval(np.squeeze(nn_retrieved['measured_pattern']),amplitude_mask,interpolate=False)

    diffraction_functions.plot_amplitude_phase_meas_retreival(matlabcdi_retrieved_interp,"matlabcdi_retrieved_interp")
    diffraction_functions.plot_amplitude_phase_meas_retreival(matlabcdi_retrieved_NOinterp,"matlabcdi_retrieved_NOinterp")

    plt.show()
    exit()

    # check if t
    matlab_complex_object=matlabcdi_retrieved["real_output"]+1j*matlabcdi_retrieved["imag_output"]

    plt.figure()
    plt.title("real retrieved complex object")
    plt.pcolormesh(np.real(matlab_complex_object))
    plt.colorbar()

    z=[]
    plt.figure()
    plt.title("interpolation points")
    plt.pcolormesh(np.real(matlab_complex_object))
    plt.colorbar()
    for _x,_y in zip(x,y):
        plt.scatter(_x,_y,s=2.0,color="red")
        z.append(matlab_complex_object[_y,_x])
    z=np.array(z)

    plt.figure()
    plt.title("interpolation grid")
    plt.scatter(x,y,c=np.real(z))
    # plt.scatter(x,y,c=np.real(z),cmap="jet")
    plt.gca().set_xlim(0,128)
    plt.gca().set_ylim(0,128)
    plt.colorbar()

    finterp_real=interpolate.interp2d(x,y,np.real(z),kind='linear')
    finterp_imag=interpolate.interp2d(x,y,np.imag(z),kind='linear')

    interp_x=np.linspace(0,128,100) # x / columns
    interp_y=np.linspace(0,128,100) # y / rows
    z_interp_real=finterp_real(
            interp_x,
            interp_y
            )
    z_interp_imag=finterp_imag(
            interp_x,
            interp_y
            )

    plt.figure()
    plt.title("AFTER interpolation")
    plt.pcolormesh(interp_x,interp_y,z_interp_real)
    plt.colorbar()

    # fig,ax=plt.subplots(1,2)
    # ax[0].pcolormesh(np.real(matlab_complex_object))
    # ax[0].set_title("real")
    # ax[1].pcolormesh(np.imag(matlab_complex_object))
    # ax[1].set_title("imag")

    # fig,ax=plt.subplots(1,2)
    # ax[0].pcolormesh(interp_x,interp_y,z_interp_real)
    # ax[0].set_title("real, INTERP")
    # ax[1].pcolormesh(interp_x,interp_y,z_interp_imag)
    # ax[1].set_title("imag, INTERP")

    plt.show()






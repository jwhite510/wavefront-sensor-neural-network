import numpy as np
import diffraction_functions
import matplotlib.pyplot as plt
import pickle
from GetMeasuredDiffractionPattern import GetMeasuredDiffractionPattern
import os


# def plot_beam(complex_b,title):
    # fig, ax = plt.subplots(2,1, figsize=(10,10))
    # fig.suptitle(title)
    # ax[0].pcolormesh(np.abs(complex_b))
    # ax[0].set_title("ABS")

def make_nice_figure(retrieved:dict):
    print(retrieved.keys())


if __name__=="__main__":
    folder_dir="nn_pictures/teslatest5_doubleksize_doublefilters_reconscostfunction_pictures/46/measured/"
    # run_name="Data_for_Jonathon_z0_1-fits_ud_1-0_reconstructed.p"
    sample_name="Data_for_Jonathon_z-500_1-fits_ud_1-0_reconstructed.p"
    filename=os.path.join(folder_dir,sample_name)
    print("filename =>", filename)

    with open(filename,"rb") as file:
        obj=pickle.load(file)
    complex_beam=np.squeeze(obj["real_output"]+1j*obj["imag_output"])


    experimental_params = {}
    experimental_params['pixel_size'] = 27e-6 # [meters] with 2x2 binning
    experimental_params['z_distance'] = 33e-3 # [meters] distance from camera
    experimental_params['wavelength'] = 13.5e-9 #[meters] wavelength
    getMeasuredDiffractionPattern = GetMeasuredDiffractionPattern(N_sim=np.shape(complex_beam)[0],
            N_meas=2,#///
            experimental_params=experimental_params)

    f_object=getMeasuredDiffractionPattern.simulation_axes['diffraction_plane']['f']

    # distance to focus
    z = 800e-6
    gamma=np.sqrt(
            1-
            (experimental_params['wavelength']*f_object.reshape(-1,1))**2-
            (experimental_params['wavelength']*f_object.reshape(1,-1))**2+
            0j # complex so it can be square rooted and imaginry
            )
    k_sq = 2 * np.pi * z / experimental_params['wavelength']
    # transfer function
    H = np.exp(1j * np.real(gamma) * k_sq) * np.exp(-1 * np.imag(gamma) * k_sq)

    complex_beam_f=np.fft.fftshift(np.fft.fft2(np.fft.fftshift(complex_beam)))
    complex_beam_f*=H
    complex_beam_prop=np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(complex_beam_f)))

    fig = diffraction_functions.plot_amplitude_phase_meas_retreival(
            {"measured_pattern":np.zeros_like(np.abs(complex_beam_prop)),
            "tf_reconstructed_diff":np.zeros_like(np.abs(complex_beam_prop)),
            "real_output":np.real(complex_beam_prop),
            "imag_output":np.imag(complex_beam_prop)},
            "z:-500 um -> + "+str(round(z*1e6,1))+" um [Simulated]",plot_spherical_aperture=True)
    # plt.savefig(str(_i)+"_SIMULATEDpropfrom-500z__"+str(round(z*1e6,1))+"_um_prop.png")
    plt.close(fig)
    print("saving:"+str(z*1e6)+"um")
    # plt.show()

    # create dictionary for publication figure
    retrieved = {}
    retrieved['300_sim'] = {
            "measured_pattern":np.zeros_like(np.abs(complex_beam_prop)),
            "tf_reconstructed_diff":np.zeros_like(np.abs(complex_beam_prop)),
            "real_output":np.real(complex_beam_prop),
            "imag_output":np.imag(complex_beam_prop)
            }


    # compare the z - 500 nm propagation to the 0 nm retrieval
    sample_name="Data_for_Jonathon_z0_1-fits_ud_1-0_reconstructed.p"
    filename=os.path.join(folder_dir,sample_name)
    with open(filename,"rb") as file:
        obj=pickle.load(file)
    fig=diffraction_functions.plot_amplitude_phase_meas_retreival(obj,"Retrieval at z=0 um")
    # plt.savefig("z0retrieval")
    plt.close(fig)
    retrieved['0'] = obj

    # plot the original retrieval at -500 nm
    sample_name="Data_for_Jonathon_z-500_1-fits_ud_1-0_reconstructed.p"
    filename=os.path.join(folder_dir,sample_name)
    with open(filename,"rb") as file:
        obj=pickle.load(file)
    fig=diffraction_functions.plot_amplitude_phase_meas_retreival(obj,"Retrieval at z=-500 um")
    # plt.savefig("z-500retrieval")
    plt.close(fig)
    retrieved['-500'] = obj

    make_nice_figure(retrieved)

    plt.show()

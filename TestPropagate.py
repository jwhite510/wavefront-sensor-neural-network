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


if __name__=="__main__":
    folder_dir="nn_pictures/14_tfprop_reconstructioncostfunction_A-6_0--S-0_5_pictures/41/measured/"
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
    x_pos = f_object * (experimental_params['wavelength'] * experimental_params['z_distance'])
    # frequency axis for propagation
    x_pos_dx=x_pos[-1]-x_pos[-2]
    N=f_object.shape[0]
    x_d_alpha=1 / (N * x_pos_dx)
    x_alpha_max=(x_d_alpha * N)/2
    x_alpha=np.arange(-x_alpha_max,x_alpha_max,x_d_alpha)

    fig = diffraction_functions.plot_amplitude_phase_meas_retreival(
            {"measured_pattern":np.zeros_like(np.abs(complex_beam)),
            "tf_reconstructed_diff":np.zeros_like(np.abs(complex_beam)),
            "real_output":np.real(complex_beam),
            "imag_output":np.imag(complex_beam)},
            "complex_beam")



    # z = 0.5 # meters distance traveled
    for z in np.arange(-2.0,2.0,0.1):
        gamma=np.sqrt(
                1-
                (experimental_params['wavelength']*x_alpha.reshape(-1,1))**2-
                (experimental_params['wavelength']*x_alpha.reshape(1,-1))**2+
                0j # complex so it can be square rooted and imaginry
                )
        k_sq = 2 * np.pi * z / experimental_params['wavelength']
        # transfer function
        H = np.exp(1j * np.real(gamma) * k_sq) * np.exp(-1 * np.imag(gamma) * k_sq)

        complex_beam_f=np.fft.fftshift(np.fft.fft(np.fft.fftshift(complex_beam)))
        complex_beam_f*=H
        complex_beam_prop=np.fft.fftshift(np.fft.ifft(np.fft.fftshift(complex_beam_f)))

        fig = diffraction_functions.plot_amplitude_phase_meas_retreival(
                {"measured_pattern":np.zeros_like(np.abs(complex_beam_prop)),
                "tf_reconstructed_diff":np.zeros_like(np.abs(complex_beam_prop)),
                "real_output":np.real(complex_beam_prop),
                "imag_output":np.imag(complex_beam_prop)},
                "complex_beam_prop:z: "+str(z))
        plt.savefig(str(z)+"_prop.png")
        plt.close(fig)
        print("saving:"+str(z))




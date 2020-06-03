import numpy as np
import diffraction_functions
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":

    folder_dir="nn_pictures/teslatest5_doubleksize_doublefilters_reconscostfunction_pictures/46/measured/"
    # run_name="Data_for_Jonathon_z0_1-fits_ud_1-0_reconstructed.p"
    sample_name="Data_for_Jonathon_z-500_1-fits_ud_1-0_reconstructed.p"
    filename=os.path.join(folder_dir,sample_name)
    print("filename =>", filename)

    with open(filename,"rb") as file:
        obj=pickle.load(file)
    # complex_beam=np.squeeze(obj["real_output"]+1j*obj["imag_output"])
    diffraction_functions.plot_amplitude_phase_meas_retreival(obj,"obj")
    plt.show()






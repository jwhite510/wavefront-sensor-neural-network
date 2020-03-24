import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy import signal
import diffraction_functions

if __name__ == "__main__":

    N = 128
    # open the object with known dimmensions
    obj_calculated_measured_axes, amplitude_mask = diffraction_functions.get_amplitude_mask_and_imagesize(N, int(N/2))

    diffraction_pattern = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(amplitude_mask)))
    # absolute value
    diffraction_pattern = np.abs(diffraction_pattern)**2

    # # # # # # # # # # # # # # #
    # open the measured data  # #
    # # # # # # # # # # # # # # #
    diffraction_calculated_measured_axes, measured_pattern = diffraction_functions.get_measured_diffraction_pattern_grid()
    df_ratio = diffraction_calculated_measured_axes['diffraction_plane']['df'] / obj_calculated_measured_axes['diffraction_plane']['df']

    diffraction_functions.plot_image_show_centroid_distance(measured_pattern, "measured_pattern", 1)
    measured_pattern = diffraction_functions.format_experimental_trace(N=N, df_ratio=df_ratio, measured_diffraction_pattern=measured_pattern, rotation_angle=3)
    diffraction_functions.plot_image_show_centroid_distance(measured_pattern, "measured_pattern", 2)
    diffraction_functions.plot_image_show_centroid_distance(diffraction_pattern, "diffraction_pattern", 3)

    plt.show()



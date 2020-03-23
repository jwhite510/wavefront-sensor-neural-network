import numpy as np
from  astropy.io import fits
import tables
import matplotlib.pyplot as plt
import pickle
from scipy import signal
from PIL import Image
import PIL.ImageOps
from generate_data import plot_complex
from scipy import signal
# import cv2
import time
import diffraction_functions


def plot_image(title, image, axes, axeslabel, scalef):
    axes = np.array(axes)
    axes *= scalef
    plt.figure()
    plt.pcolormesh(axes, axes, image)
    plt.xlabel(axeslabel)
    plt.ylabel(axeslabel)
    plt.title(title)
    plt.axvline(x=0.05)
    plt.axvline(x=-0.05)
    plt.axhline(y=0.05)
    plt.axhline(y=-0.05)


if __name__ == "__main__":

    N = 512
    # open the object with known dimmensions
    measured_axes, amplitude_mask = diffraction_functions.get_amplitude_mask_and_imagesize(N, int(N/4))

    plot_image("amplitude_mask", amplitude_mask, measured_axes['object']['x'], axeslabel="nm", scalef=1e9)

    diffraction_pattern = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(amplitude_mask)))
    # absolute value
    diffraction_pattern = np.abs(diffraction_pattern)**2

    plot_image("diffraction_pattern", diffraction_pattern, measured_axes['diffraction_plane']['f'], axeslabel="1/m *1e7", scalef=(1/1e7))

    # # # # # # # # # # # # # # #
    # open the measured data  # #
    # # # # # # # # # # # # # # #
    measured_axes, measured_pattern = diffraction_functions.get_measured_diffraction_pattern_grid()
    plot_image("measured_pattern", measured_pattern, measured_axes['diffraction_plane']['f'], axeslabel="1/m *1e7", scalef=(1/1e7))

    plt.show()



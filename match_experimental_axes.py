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

    # fits_file_name = "m3_scan_0000.fits"
    # # fits_file_name = "m3_scan_0001.fits"
    # thing = fits.open(fits_file_name)
    # measured_diffraction_pattern = thing[0].data[0,:,:]
    # print("np.shape(measured_diffraction_pattern) => ",np.shape(measured_diffraction_pattern))
    # plt.figure()
    # plt.imshow(measured_diffraction_pattern)
    # plt.show()

    # # open fits file
    # exit()

    N = 512
    # open the object with known dimmensions
    experimental_params, amplitude_mask = diffraction_functions.get_amplitude_mask_and_imagesize(N, int(N/4))

    # print("np.shape(experimental_params['object']['x']) => ",np.shape(experimental_params['object']['x']))
    # print("np.shape(experimental_params['diffraction_plane']['f']) => ",np.shape(experimental_params['diffraction_plane']['f']))

    # print("experimental_params['object']['xmax'] =>", experimental_params['object']['xmax'])
    # print("experimental_params['diffraction_plane']['fmax'] =>", experimental_params['diffraction_plane']['fmax'])

    plot_image("amplitude_mask", amplitude_mask, experimental_params['object']['x'], axeslabel="nm", scalef=1e9)

    diffraction_pattern = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(amplitude_mask)))
    # absolute value
    diffraction_pattern = np.abs(diffraction_pattern)**2

    plot_image("diffraction_pattern", diffraction_pattern, experimental_params['diffraction_plane']['f'], axeslabel="1/m *1e7", scalef=(1/1e7))

    # # # # # # # # # # # # # # #
    # open the measured data  # #
    # # # # # # # # # # # # # # #
    fits_file_name = "m3_scan_0000.fits"
    thing = fits.open(fits_file_name)
    measured_diffraction_pattern = thing[0].data[0,:,:]

    experimental_params = {}
    experimental_params['pixel_size'] = 27e-6 # [meters] with 2x2 binning
    experimental_params['z_distance'] = 33e-3 # [meters] distance from camera
    experimental_params['wavelength'] = 13.5e-9 #[meters] wavelength
    diffraction_functions.get_measured_diffraction_pattern_grid(measured_diffraction_pattern, experimental_params)

    plt.show()
    exit()



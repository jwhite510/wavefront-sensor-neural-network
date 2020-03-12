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
import cv2
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


if __name__ == "__main__":

    N = 256
    experimental_params, amplitude_mask = diffraction_functions.get_amplitude_mask_and_imagesize(N, int(N/2))

    print("np.shape(experimental_params['object']['x']) => ",np.shape(experimental_params['object']['x']))
    print("np.shape(experimental_params['diffraction_plane']['f']) => ",np.shape(experimental_params['diffraction_plane']['f']))

    print("experimental_params['object']['xmax'] =>", experimental_params['object']['xmax'])
    print("experimental_params['diffraction_plane']['fmax'] =>", experimental_params['diffraction_plane']['fmax'])

    plot_image("amplitude_mask", amplitude_mask, experimental_params['object']['x'], axeslabel="nm", scalef=1e9)

    diffraction_pattern = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(amplitude_mask)))
    # absolute value
    diffraction_pattern = np.abs(diffraction_pattern)**2

    plot_image("diffraction_pattern", diffraction_pattern, experimental_params['diffraction_plane']['f'], axeslabel="1/m", scalef=1)

    plt.show()


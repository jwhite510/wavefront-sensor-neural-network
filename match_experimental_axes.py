import numpy as np
from  astropy.io import fits
import tables
import matplotlib.pyplot as plt
from skimage.transform import resize
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

    N = 128
    # open the object with known dimmensions
    obj_calculated_measured_axes, amplitude_mask = diffraction_functions.get_amplitude_mask_and_imagesize(N, int(N/2))

    # plot_image("amplitude_mask", amplitude_mask, obj_calculated_measured_axes['object']['x'], axeslabel="nm", scalef=1e9)

    diffraction_pattern = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(amplitude_mask)))
    # absolute value
    diffraction_pattern = np.abs(diffraction_pattern)**2

    # plot_image("diffraction_pattern", diffraction_pattern, obj_calculated_measured_axes['diffraction_plane']['f'], axeslabel="1/m *1e7", scalef=(1/1e7))

    # # # # # # # # # # # # # # #
    # open the measured data  # #
    # # # # # # # # # # # # # # #
    diffraction_calculated_measured_axes, measured_pattern = diffraction_functions.get_measured_diffraction_pattern_grid()
    # plot_image("measured_pattern", measured_pattern, diffraction_calculated_measured_axes['diffraction_plane']['f'], axeslabel="1/m *1e7", scalef=(1/1e7))

    # find the ratio of the two delta f
    print("diffraction_calculated_measured_axes['diffraction_plane']['df'] =>", diffraction_calculated_measured_axes['diffraction_plane']['df'])
    print("obj_calculated_measured_axes['diffraction_plane']['df'] =>", obj_calculated_measured_axes['diffraction_plane']['df'])

    df_ratio = diffraction_calculated_measured_axes['diffraction_plane']['df'] / obj_calculated_measured_axes['diffraction_plane']['df']

    # divide scale the measured trace by this amount
    new_size = np.shape(measured_pattern)[0] * df_ratio
    new_size = int(new_size)
    measured_pattern = resize(measured_pattern, (new_size, new_size))

    print("diffraction_functions.calc_centroid(measured_pattern, axis=0) =>", diffraction_functions.calc_centroid(measured_pattern, axis=0))
    print("diffraction_functions.calc_centroid(measured_pattern, axis=1) =>", diffraction_functions.calc_centroid(measured_pattern, axis=1))

    # calculate the current centroid
    c_y = diffraction_functions.calc_centroid(measured_pattern, axis=0)
    c_x = diffraction_functions.calc_centroid(measured_pattern, axis=1)
    plt.figure()
    plt.imshow(measured_pattern)
    plt.axvline(x=c_x, color="red")
    plt.axhline(y=c_y, color="red")
    plt.title("measured_pattern resized")

    # center of the image
    plt.axvline(x=int(new_size / 2), color="green")
    plt.axhline(y=int(new_size / 2), color="green")

    # distance from centroid
    dis_x = (new_size / 2) - c_x
    dis_y = (new_size / 2) - c_y
    measured_pattern = diffraction_functions.centroid_shift(measured_pattern, dis_y, axis=0)
    measured_pattern = diffraction_functions.centroid_shift(measured_pattern, dis_x, axis=1)

    # calculate the current centroid
    c_y = diffraction_functions.calc_centroid(measured_pattern, axis=0)
    c_x = diffraction_functions.calc_centroid(measured_pattern, axis=1)
    plt.figure()
    plt.imshow(measured_pattern)
    plt.axvline(x=c_x, color="red")
    plt.axhline(y=c_y, color="red")
    plt.title("measured_pattern centroid moved")

    # crop the edges off the image
    measured_pattern = measured_pattern[int((new_size/2) - (N/2)):int((new_size/2) + (N/2)),int((new_size/2) - (N/2)):int((new_size/2) + (N/2))]

    plt.figure()
    plt.imshow(measured_pattern)
    plt.title("measured_pattern, cropped")

    plt.figure()
    plt.imshow(diffraction_pattern)
    plt.title("diffraction_pattern")

    plt.show()



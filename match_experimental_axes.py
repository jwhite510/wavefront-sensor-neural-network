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
from scipy import ndimage

def plot_image_show_centroid_distance(mat, title, figurenum):
    # calculate the current centroid
    c_y = diffraction_functions.calc_centroid(mat, axis=0)
    c_x = diffraction_functions.calc_centroid(mat, axis=1)
    plt.figure(figurenum)
    plt.imshow(mat)
    plt.axvline(x=c_x, color="yellow")
    plt.axhline(y=c_y, color="yellow")
    plt.title("measured_pattern resized")
    plt.gca().text(0.0, 0.9,"c_x: {}".format( c_x ), fontsize=10, ha='center', transform=plt.gca().transAxes, backgroundcolor="yellow")
    plt.gca().text(0.0, 0.8,"c_y: {}".format( c_y ), fontsize=10, ha='center', transform=plt.gca().transAxes, backgroundcolor="yellow")

    plt.gca().text(0.0, 0.7,"image center x: {}".format( np.shape(mat)[0] / 2 ), fontsize=10, ha='center', transform=plt.gca().transAxes, backgroundcolor="green")
    plt.gca().text(0.0, 0.6,"image center y: {}".format( np.shape(mat)[0] / 2 ), fontsize=10, ha='center', transform=plt.gca().transAxes, backgroundcolor="green")

    # center of the image
    plt.axvline(x=int(np.shape(mat)[0] / 2), color="green")
    plt.axhline(y=int(np.shape(mat)[0] / 2), color="green")

def plot_image(title, image, axes, axeslabel, scalef, fignum):
    axes = np.array(axes)
    axes *= scalef
    plt.figure(fignum)
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

    # plot_image("amplitude_mask", amplitude_mask, obj_calculated_measured_axes['object']['x'], axeslabel="nm", scalef=1e9, fignum=1)

    diffraction_pattern = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(amplitude_mask)))
    # absolute value
    diffraction_pattern = np.abs(diffraction_pattern)**2

    # plot_image("diffraction_pattern", diffraction_pattern, obj_calculated_measured_axes['diffraction_plane']['f'], axeslabel="1/m *1e7", scalef=(1/1e7), fignum=2)

    # # # # # # # # # # # # # # #
    # open the measured data  # #
    # # # # # # # # # # # # # # #
    diffraction_calculated_measured_axes, measured_pattern = diffraction_functions.get_measured_diffraction_pattern_grid()
    # plot_image("measured_pattern", measured_pattern, diffraction_calculated_measured_axes['diffraction_plane']['f'], axeslabel="1/m *1e7", scalef=(1/1e7), fignum=3)

    df_ratio = diffraction_calculated_measured_axes['diffraction_plane']['df'] / obj_calculated_measured_axes['diffraction_plane']['df']

    # divide scale the measured trace by this amount
    new_size = np.shape(measured_pattern)[0] * df_ratio
    new_size = int(new_size)
    measured_pattern = resize(measured_pattern, (new_size, new_size))

    # rotate the image by eye
    measured_pattern = ndimage.rotate(measured_pattern, 3, reshape=False)

    plot_image_show_centroid_distance(measured_pattern, "measured_pattern", 4)
    measured_pattern = diffraction_functions.center_image_at_centroid(measured_pattern)
    plot_image_show_centroid_distance(measured_pattern, "measured_pattern", 5)
    # crop the edges off the image
    measured_pattern = measured_pattern[int((new_size/2) - (N/2)):int((new_size/2) + (N/2)) , int((new_size/2) - (N/2)):int((new_size/2) + (N/2))]
    plot_image_show_centroid_distance(measured_pattern, "measured_pattern", 6)
    measured_pattern = diffraction_functions.center_image_at_centroid(measured_pattern)
    plot_image_show_centroid_distance(measured_pattern, "measured_pattern", 7)

    plt.show()



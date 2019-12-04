import numpy as np
import math
from PIL import Image, ImageDraw
from PIL import ImagePath
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import rotate
from scipy.ndimage.filters import gaussian_filter


def sum_over_all_except(mat, axis):
    """
    mat: 2 or more dimmensional numpy array
    axis: the axis to keep

    perform summation over all axes except the input argument axis
    """
    n_axes = np.shape(np.shape(mat))
    summation_axes = list(range(n_axes[0]))
    summation_axes.remove(axis)
    summation_axes = tuple(summation_axes)
    summ = np.sum(mat, axis=(summation_axes))
    return summ

def centroid_shift(mat, value, axis):
    """
    mat: the matrix which the centroid will be shifted
    value: the amount to shift the centroid
    axis: the axis along which the centroid will be shifted
    """
    # use the absolute value
    # value = -2.0
    # cast value to integer
    value = int(value)

    print("value =>", value)

    # calculate current centroid:
    start_c = calc_centroid(np.abs(mat), axis)
    target_c = start_c + value
    print("start_c =>", start_c)
    delta_c = 0

    if value > 0:
        # increase the centroid while its less than the target
        new_c = float(start_c)

        plt.ion()
        plt.figure(1)
        plt.gca().cla()
        plt.pcolormesh(np.abs(mat))
        plt.pause(0.001)

        while new_c < target_c:
            mat = np.roll(mat, shift=1, axis=axis)
            new_c = calc_centroid(np.abs(mat), axis)

            plt.figure(2)
            print("---------------")
            print("new_c =>", new_c)
            print("target_c =>", target_c)
            plt.gca().cla()
            plt.pcolormesh(np.abs(mat))
            plt.pause(0.5)

    elif value < 0:
        # decrease the centroid while its greater than the target
        new_c = float(start_c)

        plt.ion()
        plt.figure(1)
        plt.gca().cla()
        plt.pcolormesh(np.abs(mat))
        plt.pause(0.001)

        while new_c > target_c:
            mat = np.roll(mat, shift=-1, axis=axis)
            new_c = calc_centroid(np.abs(mat), axis)

            plt.figure(2)
            print("---------------")
            print("new_c =>", new_c)
            print("target_c =>", target_c)
            plt.gca().cla()
            plt.pcolormesh(np.abs(mat))
            plt.pause(0.5)

    return mat


def calc_centroid(mat, axis):
    """
    mat: 2 or more dimmensional numpy array
    axis: the axis to find the centroid
    """

    # the number of axes in the input matrix
    summ = sum_over_all_except(mat, axis)
    index_vals = np.arange(0,len(summ))

    # calculate centroid along this plot
    centroid = np.sum(summ * index_vals) / np.sum(summ)
    return centroid


def remove_ambiguitues(object):
    """
    object: 2d numpy array

    remove the translation and conjugate flip ambiguities
    of a 2d complex matrix
    """

    # plt.figure(1)
    # plt.pcolormesh(np.real(object))
    # plt.figure(2)
    # plt.pcolormesh(np.imag(object))
    # plt.figure(3)
    # plt.pcolormesh(np.abs(object))

    obj_size = np.shape(object)
    target_row = int(obj_size[0]/2)
    target_col = int(obj_size[1]/2)

    # calculate centroid along rows
    centr_row = calc_centroid(np.abs(object), axis=0)
    centr_col = calc_centroid(np.abs(object), axis=1)

    object = centroid_shift(object, value=(target_row-centr_row), axis=0)
    object = centroid_shift(object, value=(target_col-centr_col), axis=1)

    return object



def make_roll_ambiguity(object):
    n_elements = -5
    object = np.roll(object, shift=n_elements, axis=1)
    return object

def make_flip_ambiguity(object):
    object = np.flip(object, axis=1)
    object = np.flip(object, axis=0)
    # complex conjugate
    object = np.conj(object)
    return object

def make_object_phase(object, phase):
    """
        input:
        object: between 0 and 1
        phase: between 0 and 1
    """

    # multiply phase by object mask
    phase = phase * (object>0.2)

    # apply the phase
    object_with_phase = object * np.exp(-1j*phase*(2*np.pi))

    return object_with_phase

def make_object(N, min_indexes, max_indexes):
    """
        returns:
            amplitude, phase
            both with normalized values between 0 and 1

    """
    obj = np.zeros((N,N), dtype=np.complex128)

    # generate random indexes
    # np.random.seed(3367)
    indexes_n = np.random.randint(min_indexes,max_indexes)
    # for each index generate an x and y point
    x = []
    y = []
    for i in range(indexes_n):
        x.append(int(np.random.rand(1)*N))
        y.append(int(np.random.rand(1)*N))

    x.append(x[0])
    y.append(y[0])

    xy = [(x_, y_) for x_, y_ in zip(x,y)]
    image = ImagePath.Path(xy).getbbox()
    size = list(map(int, map(math.ceil, image[2:])))
    img = Image.new("RGB", [N,N], "#000000")
    img1 = ImageDraw.Draw(img)
    img1.polygon(xy, fill ="#ffffff")
    # convert to numpy array
    amplitude = np.array(img.getdata(), dtype=np.uint8).reshape(N, N, -1)
    amplitude = np.sum(amplitude, axis=2)
    amplitude = amplitude/np.max(amplitude)

    # apply gaussian filter
    amplitude = gaussian_filter(amplitude, sigma=0.8, order=0)
    # define a line with slope
    x_phase = np.linspace(-N/2, N/2, N).reshape(1,-1)
    y_phase = np.linspace(-N/2, N/2, N).reshape(-1,1)

    # create random rotation angle
    alpha_rad = np.random.rand() * 360.0
    alpha = alpha_rad*(np.pi / 180.0)
    # create random spacial frequency
    phase_frequency_min, phase_frequency_max = 0.4, 0.8
    phase_frequency = phase_frequency_min + np.random.rand() * (phase_frequency_max - phase_frequency_min)
    # rotation matrix
    x_rot = x_phase * np.cos(alpha) + y_phase * np.sin(alpha)
    y_rot = y_phase * np.cos(alpha) - x_phase * np.sin(alpha)
    z_phase_rot = np.sin(phase_frequency*x_rot)
    # make the phase between 0 and 2 pi
    z_phase_rot = z_phase_rot - np.min(z_phase_rot)
    z_phase_rot = z_phase_rot / np.max(z_phase_rot)

    # normalized phase
    # z_phase_rot = z_phase_rot * (2*np.pi)
    phase = z_phase_rot*(amplitude>0.2)

    # apply phase
    return amplitude, phase

def plot_fft(object_in):

    # diffraction pattern
    diffraction_pattern = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(object_in)))

    # plt.figure()
    fig, ax = plt.subplots(2,2, figsize=(10,10))
    fig.subplots_adjust(wspace=0.5, top=0.95, bottom=0.10)
    # object plane
    ax[0][0].pcolormesh(object_plane_x, object_plane_x, np.abs(object_in))
    ax[0][0].set_xlabel("object plane distance [m]")
    ax[0][0].set_ylabel("object plane distance [m]")
    ax[0][0].set_title("object")

    # object phase
    ax[1][0].pcolormesh(object_plane_x, object_plane_x, np.angle(object_in))
    ax[1][0].set_xlabel("object plane distance [m]")
    ax[1][0].set_ylabel("object plane distance [m]")
    ax[1][0].set_title("object phase")


    # diffraction plane
    ax[0][1].pcolormesh(diffraction_plane_x, diffraction_plane_x, np.abs(diffraction_pattern))
    ax[0][1].set_title("diffraction pattern at %i [m]" % diffraction_plane_z)
    ax[0][1].set_xlabel("diffraction plane distance [m]")
    ax[0][1].set_ylabel("diffraction plane distance [m]")

    # diffraction plane
    ax[1][1].pcolormesh(diffraction_plane_x, diffraction_plane_x, np.log10(np.abs(diffraction_pattern)))
    ax[1][1].set_title(r"$log_{10}$"+"diffraction pattern at %i [m]" % diffraction_plane_z)
    ax[1][1].set_xlabel("diffraction plane distance [m]")
    ax[1][1].set_ylabel("diffraction plane distance [m]")

if __name__ == "__main__":

    # grid space of the diffraction pattern
    N = 40 # measurement points
    diffraction_plane_x_max = 1 # meters
    diffraction_plane_z = 10 # meters
    wavelength = 400e-9

    # measured diffraction plane
    diffraction_plane_dx = 2*diffraction_plane_x_max/N
    diffraction_plane_x = diffraction_plane_dx * np.arange(-N/2, N/2, 1)

    # convert distance to frequency domain
    diffraction_plane_fx = diffraction_plane_x / (wavelength * diffraction_plane_z)
    diffraction_plane_dfx = diffraction_plane_dx / (wavelength * diffraction_plane_z)

    # x coordinates at object plane
    object_plane_dx = 1 / ( diffraction_plane_dfx * N)
    object_plane_x = object_plane_dx * np.arange(-N/2, N/2, 1)


    # construct object in the object plane
    for _ in range(100):
        object, object_phase = make_object(N, min_indexes=4, max_indexes=8)
        # object_phase = np.ones_like(object_phase)
        # construct phase
        object_with_phase = make_object_phase(object, object_phase)
        object_with_phase = remove_ambiguitues(object_with_phase)
    exit()
    remove_ambiguitues(make_roll_ambiguity(object_with_phase))

    plot_fft(object_with_phase)

    # apply roll to generate ambiguity
    plot_fft(make_roll_ambiguity(object_with_phase))

    # conjugate flip to generate ambiguity
    plot_fft(make_flip_ambiguity(object_with_phase))

    plt.show()








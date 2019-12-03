import numpy as np
import math
from PIL import Image, ImageDraw
from PIL import ImagePath
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import rotate
from scipy.ndimage.filters import gaussian_filter



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
    print("alpha =>", alpha)
    print("phase_frequency =>", phase_frequency)
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
    object, object_phase = make_object(N, min_indexes=4, max_indexes=8)
    # object_phase = np.ones_like(object_phase)
    # construct phase
    object_with_phase = make_object_phase(object, object_phase)

    plot_fft(object_with_phase)

    # apply roll to generate ambiguity
    plot_fft(make_roll_ambiguity(object_with_phase))

    # conjugate flip to generate ambiguity
    plot_fft(make_flip_ambiguity(object_with_phase))

    plt.show()








import numpy as np
import math
from PIL import Image, ImageDraw
from PIL import ImagePath
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import rotate


def make_object_phase(object, phase):
    """
        input:
        object: between 0 and 1
        phase: between 0 and 1
    """

    # multiply phase by object mask
    phase = phase * object

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
    # np.random.seed(3357)
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

    # define a line with slope
    x_phase = np.linspace(-N/2, N/2, N).reshape(1,-1)
    y_phase = np.linspace(-N/2, N/2, N).reshape(-1,1)

    # create random rotation angle
    alpha_rad = np.random.rand() * 360.0
    alpha = alpha_rad*(np.pi / 180.0)
    # create random spacial frequency
    phase_frequency_min, phase_frequency_max = 0.2, 0.8
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
    phase = z_phase_rot*amplitude

    # apply phase
    return amplitude, phase


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
    # construct phase
    object_with_phase = make_object_phase(object, object_phase)

    plt.figure(1)
    plt.pcolormesh(object_phase)
    plt.figure(2)
    plt.pcolormesh(object)

    # diffraction pattern
    diffraction_pattern = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(object)))

    # plt.figure()
    fig, ax = plt.subplots(1,2, figsize=(10,5))
    fig.subplots_adjust(wspace=0.5)
    # object plane
    ax[0].pcolormesh(object_plane_x, object_plane_x, object)
    ax[0].set_xlabel("object plane distance [m]")
    ax[0].set_ylabel("object plane distance [m]")
    ax[0].set_title("object")

    # diffraction plane
    ax[1].pcolormesh(diffraction_plane_x, diffraction_plane_x, np.abs(diffraction_pattern))
    ax[1].set_title("diffraction pattern at %i [m]" % diffraction_plane_z)
    ax[1].set_xlabel("diffraction plane distance [m]")
    ax[1].set_ylabel("diffraction plane distance [m]")
    plt.show()







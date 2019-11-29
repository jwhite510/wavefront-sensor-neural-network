import numpy as np
import math
from PIL import Image, ImageDraw
from PIL import ImagePath
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import rotate


def make_object(N, min_indexes, max_indexes):
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

    # import ipdb; ipdb.set_trace() # BREAKPOINT
    # print("BREAKPOINT")
    xy = [(x_, y_) for x_, y_ in zip(x,y)]
    image = ImagePath.Path(xy).getbbox()
    size = list(map(int, map(math.ceil, image[2:])))
    img = Image.new("RGB", [N,N], "#000000")
    img1 = ImageDraw.Draw(img)
    img1.polygon(xy, fill ="#ffffff")
    # convert to numpy array
    im_np_array = np.array(img.getdata(), dtype=np.uint8).reshape(N, N, -1)
    im_np_array = np.sum(im_np_array, axis=2)
    im_np_array = im_np_array/np.max(im_np_array)

    # random phase
    phase = np.ones((N,N))

    # random phase shift + frequency
    phi_shift = np.random.rand()*2*np.pi
    direcion = np.random.rand()
    if direcion<0.5:
        phase = phase * np.sin(phi_shift + np.linspace(0,20,N)).reshape(1,-1)
    else:
        phase = phase * np.sin(phi_shift + np.linspace(0,20,N)).reshape(-1,1)

    # apply rotation
    phase_rot = np.zeros_like(phase)
    angle = np.random.rand()*45
    phase_rot = rotate(phase, angle=angle, reshape=True)

    # get distance to crop
    if angle <= 45:
        crop = int(N*np.sin(((np.pi/180)*angle)))

    print("crop =>", crop)
    plt.figure(2)
    plt.pcolormesh(phase_rot)
    plt.figure(3)
    plt.pcolormesh(phase_rot[crop:-crop,crop:-crop])
    plt.show()
    exit()

    # apply phase
    return im_np_array


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
    object = make_object(N, min_indexes=4, max_indexes=8)

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







import numpy as np
import matplotlib.patches
import time
import pickle
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import PIL.ImageOps
import tables
import os
os.sys.path.append("../..")
import diffraction_functions



def plot(array):
    plt.figure()
    plt.imshow(array)
    # print(array)
    plt.show()

def plot_complex(array):
    fig = plt.figure(figsize=(10,4))
    gs = fig.add_gridspec(1,3)

    ax = fig.add_subplot(gs[0,0])
    im = ax.imshow(np.angle(array))
    fig.colorbar(im, ax=ax)
    ax.set_title("angle")

    ax = fig.add_subplot(gs[0,1])
    im = ax.imshow(np.real(array))
    fig.colorbar(im, ax=ax)
    ax.set_title("real")

    ax = fig.add_subplot(gs[0,2])
    im = ax.imshow(np.imag(array))
    fig.colorbar(im, ax=ax)
    ax.set_title("imag")

    # plt.show()

def plot_complex_phaseplots(array):
    try:
        fig = plt.figure(figsize=(18,4))
        gs = fig.add_gridspec(1,6)

        ax = fig.add_subplot(gs[0,0])
        im = ax.imshow(np.angle(array))
        fig.colorbar(im, ax=ax)
        ax.set_title("angle")

        ax = fig.add_subplot(gs[0,1])
        im = ax.imshow(np.real(array))
        fig.colorbar(im, ax=ax)
        ax.set_title("real")
        ax.axhline(y=np.shape(array)[0]/2, color="blue", alpha=0.5)
        ax.axvline(x=np.shape(array)[1]/2, color="blue", alpha=0.5)

        ax = fig.add_subplot(gs[0,2])
        im = ax.imshow(np.imag(array))
        fig.colorbar(im, ax=ax)
        ax.set_title("imag")
        ax.axhline(y=np.shape(array)[0]/2, color="red", alpha=0.5)
        ax.axvline(x=np.shape(array)[1]/2, color="red", alpha=0.5)

        ax = fig.add_subplot(gs[0,3])
        im = ax.imshow(np.abs(array))
        fig.colorbar(im, ax=ax)
        ax.set_title("abs")

        ax = fig.add_subplot(gs[0,4])
        center_row=np.shape(array)[0]
        center_col=np.shape(array)[1]
        ax.plot(np.real(array[int(center_row/2),:]), color="blue")
        ax.plot(np.imag(array[int(center_row/2),:]), color="red")
        ax.axvline(x=np.shape(array)[0]/2, color="black", alpha=0.5)
        ax.axhline(y=0, color="black", alpha=0.5)

        ax = fig.add_subplot(gs[0,5])
        center_row=np.shape(array)[0]
        center_col=np.shape(array)[1]
        ax.plot(np.real(array[:,int(center_col/2)]), color="blue")
        ax.plot(np.imag(array[:,int(center_col/2)]), color="red")
        ax.axvline(x=np.shape(array)[1]/2, color="black", alpha=0.5)
        ax.axhline(y=0, color="black", alpha=0.5)

        # plt.show()
    except Exception as e:
        print("DANGER WILL ROBINSON")
        print(e)

def plot_zernike(array):

    try:
        print("array.shape =>", array.shape)

        # difficulty number
        minval = 0.2
        # image_relative_size can be between minvaland 1
        image_relative_size = minval + np.random.rand()*(1-minval)
        print("image_relative_size =>", image_relative_size)
        # image_relative_size = np.random.rand()
        # max x, y location:
        max_xy_location = 1.0 - image_relative_size;
        # x, y start is between these two values
        x_tl = np.random.rand()*max_xy_location
        x_br = x_tl + image_relative_size;
        y_tl = np.random.rand()*max_xy_location;
        y_br = y_tl + image_relative_size;

        x_tl *=2; x_tl -=1; # // shift to -1. 1 coordinates
        x_br *=2; x_br -=1; # // shift to -1. 1 coordinates
        y_tl *=2; y_tl -=1; # // shift to -1. 1 coordinates
        y_br *=2; y_br -=1; # // shift to -1. 1 coordinates

        crop_size = 200
        array = array[int(array.shape[0]/2 - crop_size/2) : int(array.shape[0]/2 + crop_size/2),
                    int(array.shape[1]/2 - crop_size/2) : int(array.shape[1]/2 + crop_size/2)]

        # crop the array

        plt.figure()
        plt.pcolormesh(
                np.linspace(-1,1,np.shape(array)[0]), # x value
                np.linspace(-1,1,np.shape(array)[1]), # y value
                np.abs(array)
                )

        rect = matplotlib.patches.Rectangle((-1.0,1.0), 2.0, -2.0,
                linewidth=5, edgecolor="r", facecolor='none')
        plt.gca().add_patch(rect)

        print("x_tl =>", x_tl)
        print("y_tl =>", y_tl)

        size_x = x_br - x_tl
        size_y = y_br - y_tl
        print("size_x =>", size_x)
        print("size_y =>", size_y)

        rect = matplotlib.patches.Rectangle((x_tl,y_tl), size_x, size_y,
                linewidth=5, edgecolor="orange", facecolor='none')
        plt.gca().add_patch(rect)

        fign = 5
        plt.savefig("./LOW_scale_crop_difficulty_{}.png".format(str(fign)))

        # plt.axhline(y=y_tl, color="blue")
        # plt.axvline(x=x_tl, color="blue")


    except Exception as e:
        print(e)


def plot_complex_diffraction(array):
    array = array.astype(np.complex128)
    print("plot_complex_diffraction")
    fig = plt.figure(figsize=(10,7))
    gs = fig.add_gridspec(2,3)

    ax = fig.add_subplot(gs[0,0])
    im = ax.imshow(np.angle(array))
    fig.colorbar(im, ax=ax)
    ax.set_title("angle")

    ax = fig.add_subplot(gs[0,1])
    im = ax.imshow(np.real(array))
    fig.colorbar(im, ax=ax)
    ax.set_title("real")

    ax = fig.add_subplot(gs[0,2])
    im = ax.imshow(np.imag(array))
    fig.colorbar(im, ax=ax)
    ax.set_title("imag")

    # make the intensity pattern
    ax = fig.add_subplot(gs[1,0])
    im = ax.imshow(np.abs(array)**2)
    fig.colorbar(im, ax=ax)
    ax.set_title("intensity")

    # make the diffraction pattern
    ax = fig.add_subplot(gs[1,1])
    diffraction = np.abs(np.fft.fftshift(np.fft.fft2(np.fft.fftshift(array))))**2
    im = ax.imshow(diffraction)
    fig.colorbar(im, ax=ax)
    ax.set_title("diffraction")

def fft_ifft(array):

    plot_complex_diffraction(array)
    array = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(array)))
    array = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(array)))
    plot_complex_diffraction(array)

def save_interped_arr(array):
    print(np.shape(array))
    with open("interped_arr.p", "wb") as file:
        pickle.dump(array, file)

def save_slice(array):
    with open("slice.p", "wb") as file:
        pickle.dump(array, file)

def save_f(array):
    with open("f.p", "wb") as file:
        pickle.dump(array, file)

def fftshift2(array):
    array = np.fft.fftshift(array)

    fig = plt.figure(figsize=(10,4))
    gs = fig.add_gridspec(1,3)

    ax = fig.add_subplot(gs[0,0])
    ax.imshow(np.angle(array))
    ax.set_title("np fftshift angle")

    ax = fig.add_subplot(gs[0,1])
    ax.imshow(np.real(array))
    ax.set_title("np fftshift real")

    ax = fig.add_subplot(gs[0,2])
    ax.imshow(np.imag(array))
    ax.set_title("np fftshift imag")

    # plt.show()

def fft2(array):
    array = np.fft.fftshift(array)
    array = np.fft.fft2(array)
    array = np.fft.fftshift(array)

    fig = plt.figure(figsize=(10,4))
    gs = fig.add_gridspec(1,3)

    ax = fig.add_subplot(gs[0,0])
    ax.imshow(np.angle(array))
    ax.set_title("np fft2 angle")

    ax = fig.add_subplot(gs[0,1])
    ax.imshow(np.real(array))
    ax.set_title("np fft2 real")

    ax = fig.add_subplot(gs[0,2])
    ax.imshow(np.imag(array))
    ax.set_title("np fft2 imag")

    # plt.show()

def get_wavefront_sensor(a):
    N = 128
    _, b = diffraction_functions.get_amplitude_mask_and_imagesize(N, int(N/2))
    b = b.astype(np.float32)
    return b

def get_wavefront_sensor_f(a):
    N = 128
    measured_axes, b = diffraction_functions.get_amplitude_mask_and_imagesize(N, int(N/2))
    # b = b.astype(np.float32)
    b = measured_axes["diffraction_plane"]["f"].astype(np.float32)
    return b

def show(a):
    print("called me")
    plt.show()

def savepickle(filename, array_in):
    with open(filename, "wb") as file:
        pickle.dump(array_in, file)

def string_and_array(string, array):
    print("string =>", string)
    print("np.shape(array) => ",np.shape(array))

def array_only(array):
    print("np.shape(array) => ",np.shape(array))

def get_shape(array):
    print("np.shape(array) => ",np.shape(array))

    plt.figure()
    plt.imshow(array[0,:,:])

    plt.figure()
    plt.imshow(array[1,:,:])

    plt.figure()
    plt.imshow(array[2,:,:])

    plt.figure()
    plt.imshow(array[3,:,:])

    plt.figure()
    plt.imshow(array[4,:,:])

    plt.figure()
    plt.imshow(array[5,:,:])

    plt.figure()
    plt.imshow(array[6,:,:])

    plt.figure()
    plt.imshow(array[7,:,:])

    plt.figure()
    plt.imshow(array[8,:,:])

    plt.figure()
    plt.imshow(array[9,:,:])

    plt.figure()
    plt.imshow(array[10,:,:])

    plt.figure()
    plt.imshow(array[11,:,:])

    plt.show()

def create_dataset(filename):

    print("called create_dataset")
    print(filename)
    N = 128
    with tables.open_file(filename, "w") as hdf5file:

        # create array for the object
        hdf5file.create_earray(hdf5file.root, "object_real", tables.Float32Atom(), shape=(0,N*N))

        # create array for the object phase
        hdf5file.create_earray(hdf5file.root, "object_imag", tables.Float32Atom(), shape=(0,N*N))

        # create array for the image
        hdf5file.create_earray(hdf5file.root, "diffraction", tables.Float32Atom(), shape=(0,N*N))

        hdf5file.create_earray(hdf5file.root, "N", tables.Int32Atom(), shape=(0,1))

        hdf5file.close()

    with tables.open_file(filename, mode='a') as hd5file:
        # save the dimmensions of the data
        hd5file.root.N.append(np.array([[N]]))

def write_to_dataset(filename, array):
    # print("writing dataset")

    if True in np.isnan(array):
        print("complex_object is NAN!!!!!!!")
        print("continuing")
        return

    if True in np.isinf(array):
        print("complex_object is inf!!!!!!!")
        print("continuing")
        return

    # normalize the array
    array = array / np.max(np.abs(array))

    object_real = np.real(array)
    object_imag = np.imag(array)
    diffraction_pattern_with_noise = np.abs(np.fft.fftshift(np.fft.fft2(np.fft.fftshift(array))))**2

    # normalize
    diffraction_pattern_with_noise = diffraction_pattern_with_noise / np.max(diffraction_pattern_with_noise)
    diffraction_pattern_with_noise = diffraction_functions.center_image_at_centroid(diffraction_pattern_with_noise)

    # plt.figure()
    # plt.imshow(object_real)

    # plt.figure()
    # plt.imshow(object_imag)

    # plt.figure()
    # plt.imshow(diffraction_pattern_with_noise)

    # plt.show()

    with tables.open_file(filename, mode='a') as hd5file:
        hd5file.root.object_real.append(object_real.reshape(1,-1))
        hd5file.root.object_imag.append(object_imag.reshape(1,-1))
        hd5file.root.diffraction.append(diffraction_pattern_with_noise.reshape(1,-1))

    # print("wrote to dataset!")


def view_array(array):
    print("np.shape(array) => ",np.shape(array))
    # save all of these elements to the hdf5 file
    plt.figure()
    plt.imshow(np.abs(array[0,:,:]))
    plt.figure()
    plt.imshow(np.abs(array[1,:,:]))
    plt.figure()
    plt.imshow(np.abs(array[2,:,:]))
    plt.show()

def save_to_hdf5(filename, wavefront_sensor, wavefront):

    # plt.figure()
    # plt.imshow(np.real(wavefront[0,:,:]))
    # plt.title("wavefront")
    # plt.colorbar()

    # plt.figure()
    # plt.imshow(np.real(wavefront_sensor[0,:,:]))
    # plt.title("wavefront_sensor")
    # plt.colorbar()
    # plt.show()
    print("save_to_hdf5 called")
    try:

        if True in np.isnan(wavefront):
            print("complex_object is NAN!!!!!!!")
            return

        if True in np.isinf(wavefront):
            print("complex_object is inf!!!!!!!")
            return

        with tables.open_file(filename, mode='a') as hd5file:
            for i in range(np.shape(wavefront)[0]):
                object_real = np.real(wavefront[i,:,:])
                object_imag = np.imag(wavefront[i,:,:])
                diffraction_pattern_with_noise = np.abs(np.fft.fftshift(np.fft.fft2(np.fft.fftshift(wavefront_sensor[i,:,:]))))**2

                # normalize
                diffraction_pattern_with_noise = diffraction_pattern_with_noise / np.max(diffraction_pattern_with_noise)
                diffraction_pattern_with_noise = diffraction_functions.center_image_at_centroid(diffraction_pattern_with_noise)
                hd5file.root.object_real.append(object_real.reshape(1,-1))
                hd5file.root.object_imag.append(object_imag.reshape(1,-1))
                hd5file.root.diffraction.append(diffraction_pattern_with_noise.reshape(1,-1))

            print("calling flush")
            hd5file.flush()

    except Exception as e:
        print(e)

def testcall2(filename, array1, array2):
    print("filename =>", filename)
    print("np.shape(array2) => ",np.shape(array2))
    print("np.shape(array1) => ",np.shape(array1))
    print("testcall2 called")
    print("in python")


if __name__ == "__main__":
    print("running main!!!")
    # create_dataset("train.hdf5")
    # test open the data set
    # index = 19
    for index in range(30000, 30005):
        print("opening index:" + str(index))
        with tables.open_file("train.hdf5", mode="r") as hdf5file:

            # print("hdf5file.root.N =>", hdf5file.root.N[0,0])
            N = hdf5file.root.N[0,0]

            object_real = hdf5file.root.object_real[index,:].reshape(N,N)
            object_imag = hdf5file.root.object_imag[index,:].reshape(N,N)
            diffraction = hdf5file.root.diffraction[index,:].reshape(N,N)

        print("np.shape(diffraction) => ",np.shape(diffraction))

        # plt.figure(2)
        # plt.pcolormesh(object_real)

        # plt.figure(3)
        # plt.pcolormesh(object_imag)

        # plt.figure(4)
        # plt.pcolormesh(diffraction)
        # plt.show()


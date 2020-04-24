import numpy as np
import time
import pickle
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import PIL.ImageOps
import diffraction_functions
import tables



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

def save_to_hdf5(filename, array):
    try:

        if True in np.isnan(array):
            print("complex_object is NAN!!!!!!!")
            print("continuing")
            return

        if True in np.isinf(array):
            print("complex_object is inf!!!!!!!")
            print("continuing")
            return

        with tables.open_file(filename, mode='a') as hd5file:
            for i in range(np.shape(array)[0]):
                object_real = np.real(array[i,:,:])
                object_imag = np.imag(array[i,:,:])
                diffraction_pattern_with_noise = np.abs(np.fft.fftshift(np.fft.fft2(np.fft.fftshift(array[i,:,:]))))**2

                # normalize
                diffraction_pattern_with_noise = diffraction_pattern_with_noise / np.max(diffraction_pattern_with_noise)
                diffraction_pattern_with_noise = diffraction_functions.center_image_at_centroid(diffraction_pattern_with_noise)
                hd5file.root.object_real.append(object_real.reshape(1,-1))
                hd5file.root.object_imag.append(object_imag.reshape(1,-1))
                hd5file.root.diffraction.append(diffraction_pattern_with_noise.reshape(1,-1))

                # plt.figure()
                # plt.title(str(i))
                # plt.imshow(object_real)
                # plt.figure()
                # plt.title(str(i))
                # plt.imshow(diffraction_pattern_with_noise)
                # plt.show()

    except Exception as e:
        print(e)


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


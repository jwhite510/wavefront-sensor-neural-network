import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tables
import diffraction_functions
import os
import random
from PIL import Image
import PIL.ImageOps

def plot_sample(N, object_phase, object_amplitude, diffraction_pattern):

    print("object_phase[int(N/2) , int(N/2) ] =>", object_phase[int(N/2) , int(N/2) ])

    fig = plt.figure(1, figsize=(5,10))
    fig.clf()
    gs = fig.add_gridspec(3,1)

    ax = fig.add_subplot(gs[0,0])
    im = ax.imshow(object_phase)
    ax.set_title("object_phase")
    cax = fig.add_axes([0.8, 0.65, 0.05, 0.2])
    fig.colorbar(im, cax=cax, orientation='vertical')

    ax = fig.add_subplot(gs[1,0])
    im = ax.imshow(object_amplitude)
    ax.set_title("object_amplitude")
    cax = fig.add_axes([0.8, 0.39, 0.05, 0.2])
    fig.colorbar(im, cax=cax, orientation='vertical')

    ax = fig.add_subplot(gs[2,0])
    im = ax.imshow(diffraction_pattern)
    ax.set_title("diffraction_pattern")
    cax = fig.add_axes([0.8, 0.12, 0.05, 0.2])
    fig.colorbar(im, cax=cax, orientation='vertical')


def print_debug_variables(debug_locals):

    debug_x = None
    from types import ModuleType
    print("(variable):"+19*" "+"(type):"+23*" "+"(shape):")
    for debug_x in debug_locals:
        if debug_x[0:2] != "__" and not callable(debug_locals[debug_x]) and not isinstance(debug_locals[debug_x], ModuleType) and debug_x!="debug_x" and debug_x!="debug_locals":
            print(debug_x, end='')
            print((30-len(debug_x))*' ', end='')
            print(type(debug_locals[debug_x]), end='')
            print((30-len(str(type(debug_locals[debug_x]))))*' ', end='')
            try:
                print(np.shape(np.array(debug_locals[debug_x])), end='')
            except:
                pass
            print("")
    print("")


def make_wavefront_sensor_image(N):

    assert N==128

    def plot_zernike(N,m,n):
        zernike = diffraction_functions.zernike_polynomial(N,m,n)
        plt.figure()
        plt.pcolormesh(zernike, cmap="jet")
        plt.title("m:"+str(m)+" n:"+str(n))

    # get the png image for amplitude
    im = Image.open("size_6um_pitch_600nm_diameter_300nm_psize_5nm.png")
    im = PIL.ImageOps.invert(im)
    im = im.resize((64,64))
    amplitude = np.array(im.getdata(), dtype=np.uint8).reshape(im.size[0], im.size[1], -1)
    amplitude = np.sum(amplitude, axis=2)

    # pad the amplitude image with zeros
    amplitude = np.concatenate((amplitude, np.zeros((64,32))), axis=1)
    amplitude = np.concatenate((np.zeros((64,32)), amplitude), axis=1)
    amplitude = np.concatenate((np.zeros((32,128)), amplitude), axis=0)
    amplitude = np.concatenate((amplitude, np.zeros((32,128))), axis=0)
    amplitude *= 1/np.max(amplitude) # normalize
    # amplitude[amplitude>0.5] = 1
    # concat 32

    zernike_coefficients = [
            #(m,n)
            (1,1),
            (-1,1),
            (-2,2),
            (0,2),
            (2,2),
            (-3,3),
            (-1,3),
            (1,3),
            (3,3),
            ]
    zernike_phase = np.zeros((N,N))
    for z_coefs in zernike_coefficients:
        zernike_phase += np.random.rand()*diffraction_functions.zernike_polynomial(N,z_coefs[0],z_coefs[1])

    # normalize between -pi and +pi
    zernike_phase -= np.min(zernike_phase)
    zernike_phase *= 1/np.max(zernike_phase)
    zernike_phase *= (2*np.pi)
    zernike_phase -= (np.pi)

    # plt.figure()
    # plt.pcolormesh(amplitude, cmap="jet")
    # plt.colorbar()

    # plt.figure()
    # plt.pcolormesh(zernike_phase, cmap="jet")
    # plt.colorbar()

    return zernike_phase, amplitude

def make_simulated_object(N, min_indexes, max_indexes):

            # create object
            object_amplitude = diffraction_functions.make_object(N=N, min_indexes=min_indexes, max_indexes=max_indexes)

            # center the object and remove ambiguity
            object_amplitude = diffraction_functions.remove_ambiguitues(object_amplitude)
            object_amplitude = diffraction_functions.remove_ambiguitues(object_amplitude)

            # make sure the object is normalized
            object_amplitude = object_amplitude - np.min(object_amplitude)
            object_amplitude = object_amplitude / np.max(object_amplitude)

            # generate phase to apply to the object
            object_phase = diffraction_functions.create_phase(N) # between -pi and +pi

            return object_phase, object_amplitude

def retrieve_coco_image(N, path, scale):
    """
    N: resolution of output image
    path: path to coco dataset image folder
    scale: scaling to apply to image (zoom in to decrease high frequencies)
    """
    coc_images = os.listdir(path)

    amplitude_filepath = os.path.join(path, random.choice(coc_images))
    amplitude_im = Image.open(amplitude_filepath).convert("LA")
    amplitude_im = diffraction_functions.rescale_image(amplitude_im, scale)
    amplitude_im = amplitude_im.resize((N,N))
    amplitude_im = np.array(amplitude_im)
    amplitude_im = amplitude_im[:,:,0]

    phase_filepath = os.path.join(path, random.choice(coc_images))
    phase_im = Image.open(phase_filepath).convert("LA")
    phase_im = diffraction_functions.rescale_image(phase_im, scale)
    phase_im = phase_im.resize((N,N))
    phase_im = np.array(phase_im)
    phase_im = phase_im[:,:,0]

    amplitude_im -= np.min(amplitude_im)
    amplitude_im = amplitude_im / np.max(amplitude_im)

    phase_im -= np.min(phase_im)
    phase_im = phase_im /  np.max(phase_im)
    phase_im = phase_im * 2
    phase_im = phase_im - 1
    phase_im *= np.pi

    return phase_im, amplitude_im



def make_dataset(filename, N, samples):
    # create the tables file

    with tables.open_file(filename, "w") as hdf5file:

        # create array for the object
        hdf5file.create_earray(hdf5file.root, "object_amplitude", tables.Float64Atom(), shape=(0,N*N))

        # create array for the object phase
        hdf5file.create_earray(hdf5file.root, "object_phase", tables.Float64Atom(), shape=(0,N*N))

        # create array for the image
        hdf5file.create_earray(hdf5file.root, "diffraction", tables.Float64Atom(), shape=(0,N*N))

        hdf5file.create_earray(hdf5file.root, "N", tables.Int32Atom(), shape=(0,1))

        hdf5file.close()


    with tables.open_file(filename, mode='a') as hd5file:

        # save the dimmensions of the data
        hd5file.root.N.append(np.array([[N]]))
        # plt.ion()
        for i in range(samples):

            if i % 100 == 0:
                print("Generating sample %i of %i" % (i, samples))

            def plot_thing(arr, num, title, range=None):
                arr = np.array(arr)
                # make the center visible
                arr[int(N/2), int(N/2)] = np.max(arr)
                if range:
                    arr[0,0] = range[0]
                    arr[0,1] = range[1]
                plt.figure(num)
                plt.imshow(arr)
                plt.colorbar()
                plt.title(title)
                plt.savefig("./"+str(num))
                # os.system("display "+str(num)+".png & disown")

            # object_phase, object_amplitude = make_simulated_object(N, min_indexes=4, max_indexes=8)
            # plot_thing(object_phase, 1)
            # plot_thing(object_amplitude, 2)

            # object_phase, object_amplitude = make_wavefront_sensor_image(N)
            # plot_thing(object_phase, 2)
            # plot_thing(object_amplitude, 3)

            object_phase, object_amplitude = retrieve_coco_image(N, "./coco_dataset/val2014/", scale=1.0)
            # plot_thing(object_phase, 4, "object_phase")

            # set phase at center to 0 (introduces phase discontinuity)
            object_phase-=object_phase[int(N/2), int(N/2)]
            object_phase += np.pi
            object_phase = np.mod(object_phase, 2*np.pi)
            object_phase -= np.pi

            # circular crop the phase
            diffraction_functions.circular_crop(object_phase, 0.3)
            diffraction_functions.circular_crop(object_amplitude, 0.3)

            complex_object = object_amplitude * np.exp(1j * object_phase)

            #TODO: decide to do this or not
            # set phase at center to 0
            # phase_at_center = np.angle(complex_object)[int(N/2), int(N/2)]
            # complex_object *= np.exp(-1j*phase_at_center)

            """
                    reduce parts of object below threshold
            """
            # complex_object[np.abs(complex_object)<0.01] = 0

            """
                    crop the complex_object in a circle
            """
            # diffraction_functions.circular_crop(complex_object, 0.3)

            """
                    normalize amplitude
            """
            complex_object *= 1 / np.max(np.abs(complex_object))

            # set the phase between 0:(0 pi) and 1:(2 pi)
            # object_phase = np.angle(complex_object)
            object_amplitude = np.abs(complex_object)
            # object_phase[int(N/2), int(N/2)] = -np.pi # make sure the center is 0, it might be 1 (0 or 2pi)

            # plot_thing(object_phase, 4, "object_phase")

            # set the phase between 0 and 1
            object_phase += np.pi
            object_phase /= 2*np.pi

            diffraction_pattern = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(complex_object)))
            # absolute value
            diffraction_pattern = np.abs(diffraction_pattern)
            # normalize the diffraction pattern
            diffraction_pattern = diffraction_pattern / np.max(diffraction_pattern)

            # TODO: maybe not normalize diffraction_pattern?

            # # verify phase is 0pi (0.5) at center
            # plt.figure(19)
            # plt.plot(object_phase[int(N/2),:])
            # plt.plot([0,N], [0.5, 0.5])
            # plt.plot([N/2,N/2], [0, 1])

            if i % 100 == 0:
                plot_sample(N, object_phase, object_amplitude, diffraction_pattern)
                plt.pause(0.001)

            hd5file.root.object_amplitude.append(object_amplitude.reshape(1,-1))
            hd5file.root.object_phase.append(object_phase.reshape(1,-1))
            hd5file.root.diffraction.append(diffraction_pattern.reshape(1,-1))

            # # reconstruct diffraction pattern
            # recons_diff = diffraction_functions.construct_diffraction_pattern(object_amplitude, object_phase)
            # plt.figure()
            # plt.imshow(recons_diff)
            # plt.colorbar()
            # plt.savefig("./4.png")
            # os.system("display 4.png & disown")
            # exit()


if __name__ == "__main__":
    # generate a data set
    N = 128

    make_dataset("train_data.hdf5", N=N, samples=1000)

    make_dataset("test_data.hdf5", N=N, samples=200)

    # test open the data set
    index = 4
    with tables.open_file("train_data.hdf5", mode="r") as hdf5file:

        # print("hdf5file.root.N =>", hdf5file.root.N[0,0])
        N = hdf5file.root.N[0,0]

        object = hdf5file.root.object_amplitude[index,:].reshape(N,N)
        object_phase = hdf5file.root.object_phase[index,:].reshape(N,N)
        diffraction = hdf5file.root.diffraction[index,:].reshape(N,N)

        # plt.figure(2)
        # plt.pcolormesh(object)
        # plt.figure(3)
        # plt.pcolormesh(object_phase)
        # plt.figure(4)
        # plt.pcolormesh(diffraction)
        # plt.show()



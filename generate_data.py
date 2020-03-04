import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tables
import diffraction_functions
from skimage.restoration import unwrap_phase
from skimage.transform import resize
import os
import random
from PIL import Image
import imageio
import tensorflow as tf
import PIL.ImageOps
import time

def plot_complex(title, complex_array, num, plot_z_radius=False, zoom_in=None, axis_limit=None):
    """
        zoom_in: a float specifying the fraction of the peak value of absolute value to set the limits
    """
    complex_array_intg = np.sum(np.abs(complex_array), axis=0)
    if zoom_in is not None:
        axis_limit = np.max(complex_array_intg) * zoom_in
        i = int(len(complex_array_intg) / 2)
        j = i
        di = None
        while complex_array_intg[i] > axis_limit:
            i+=1
            di = i - j

            if i > 1020:
                break

    if axis_limit is not None:
        ax_min = (np.shape(complex_array)[0] / 2) - axis_limit
        ax_max = (np.shape(complex_array)[0] / 2) + axis_limit

    assert isinstance(title, str)
    assert isinstance(complex_array, np.ndarray)
    # plt.figure(num)
    # plt.clf()
    fig, ax = plt.subplots(1,4, figsize=(15,5), num=num)
    fig.figsize = (15,5)
    fig.text(0.5, 0.90,title, ha="center", size=30)
    im = ax[0].imshow(np.abs(complex_array))
    ax[0].set_title("abs")
    fig.colorbar(im, ax=ax[0])
    if zoom_in is not None:
        ax[0].set_xlim(j - di, j + di)
        ax[0].set_ylim(j - di, j + di)
    elif axis_limit is not None:
        ax[0].set_xlim(ax_min, ax_max)
        ax[0].set_ylim(ax_min, ax_max)

    im = ax[1].imshow(np.real(complex_array))
    ax[1].set_title("real")
    fig.colorbar(im, ax=ax[1])
    if zoom_in is not None:
        ax[1].set_xlim(j - di, j + di)
        ax[1].set_ylim(j - di, j + di)
    elif axis_limit is not None:
        ax[1].set_xlim(ax_min, ax_max)
        ax[1].set_ylim(ax_min, ax_max)

    im = ax[2].imshow(np.imag(complex_array))
    ax[2].set_title("imag")
    fig.colorbar(im, ax=ax[2])
    if zoom_in is not None:
        ax[2].set_xlim(j - di, j + di)
        ax[2].set_ylim(j - di, j + di)
    elif axis_limit is not None:
        ax[2].set_xlim(ax_min, ax_max)
        ax[2].set_ylim(ax_min, ax_max)

    # unwrapped_phase = unwrap_phase(np.angle(complex_array))
    unwrapped_phase = np.angle(complex_array)
    # unwrapped_phase[np.abs(complex_array)<0.01] = 0

    if plot_z_radius:
        unwrapped_phase[z_radius>1] = unwrapped_phase[int(N_zernike/2), int(N_zernike/2)]

    im = ax[3].imshow(unwrapped_phase)
    ax[3].set_title("angle")
    fig.colorbar(im, ax=ax[3])
    if zoom_in is not None:
        ax[3].set_xlim(j - di, j + di)
        ax[3].set_ylim(j - di, j + di)
    elif axis_limit is not None:
        ax[3].set_xlim(ax_min, ax_max)
        ax[3].set_ylim(ax_min, ax_max)

    return fig

def tf_factorial(x):
    return tf.exp(tf.lgamma(x + 1)) # factorial of x

def build_tf_propagate_gaussian_graph(N_zernike, scalef):

    x = np.linspace(-1,1,N_zernike).reshape(-1,1)
    y = np.linspace(-1,1,N_zernike).reshape(1,-1)

    dx = x[1,0] - x[0,0]
    dy = y[0,1] - y[0,0]

    w_x = 0.1 *scalef
    w_y = 0.1 *scalef
    s_x = 0 + (dx)*0.5 # so the linear phase is 0
    s_y = 0 + (dy)*0.5 # so the linear phase is 0

    z = np.exp((-(x-s_x)**2)/w_x**2) * np.exp((-(y-s_y)**2)/w_y**2)

    # zernike_phase # place holder
    zernike_phase = tf.placeholder(tf.float32, shape=[N_zernike , N_zernike])
    zernike_phase_complex = tf.complex(imag=zernike_phase, real=tf.zeros_like(zernike_phase))
    z_compex = z * tf.exp(zernike_phase_complex)

    prop = tf.manip.roll(z_compex, shift=int(N_zernike/2), axis=0)
    prop = tf.manip.roll(prop, shift=int(N_zernike/2), axis=1)
    prop = tf.fft2d(prop)
    prop = tf.manip.roll(prop, shift=int(N_zernike/2), axis=0)
    prop = tf.manip.roll(prop, shift=int(N_zernike/2), axis=1)

    nodes = {}
    nodes["z_compex"] = z_compex
    nodes["zernike_phase"] = zernike_phase
    nodes["prop"] = prop

    return nodes

def build_tf_zernike_graph(N_zernike, scalef):

    m_ph = tf.placeholder(tf.float64, shape=[]) # placeholder
    n_ph = tf.placeholder(tf.float64, shape=[]) # placeholder

    scale = 10.0 / scalef
    x = np.linspace(-1*scale,1*scale,N_zernike).reshape(1,-1)
    y = np.linspace(-1*scale,1*scale,N_zernike).reshape(-1,1)

    rho = np.sqrt(x**2 + y**2)
    rho = np.expand_dims(rho, axis=2)

    phi = np.arctan2(y,x)

    k = tf.range(0, ((n_ph-m_ph)/2)+1)
    k = tf.reshape(k, [1,1,-1])
    numerator = (-1)**k
    numerator *= tf_factorial(n_ph-k)

    denominator = tf_factorial(k)
    denominator *= tf_factorial(((n_ph+m_ph)/2)-k)
    denominator *= tf_factorial(((n_ph-m_ph)/2)-k)

    R = (numerator / denominator)*rho**(n_ph-2*k)
    R = tf.reduce_sum(R, axis=2)
    # R = np.sum(R, axis=2)
    Z_even = R*tf.cos(m_ph*phi)
    Z_odd = R*tf.sin(m_ph*phi)

    r = np.sqrt(x**2 + y**2)
    nodes = {}
    nodes["m_ph"] = m_ph
    nodes["n_ph"] = n_ph
    nodes["Z_even"] = Z_even
    nodes["Z_odd"] = Z_odd

    return nodes, r


def save_gif_image(figure1, figure2, gif_images):

        # save image to a gif
        figure1.canvas.draw()
        figure2.canvas.draw()
        image3 = np.frombuffer(figure2.canvas.tostring_rgb(), dtype='uint8')
        image3  = image3.reshape(figure2.canvas.get_width_height()[::-1] + (3,))
        image1 = np.frombuffer(figure1.canvas.tostring_rgb(), dtype='uint8')
        image1  = image1.reshape(figure1.canvas.get_width_height()[::-1] + (3,))
        both_images = np.concatenate((image1, image3), axis=0)
        gif_images.append(both_images)

        return gif_images


def plot_sample(N, object_real, object_imag, diffraction_pattern):


    fig = plt.figure(1, figsize=(5,10))
    fig.clf()
    gs = fig.add_gridspec(3,1)

    ax = fig.add_subplot(gs[0,0])
    im = ax.imshow(object_real)
    ax.set_title("object_real")
    cax = fig.add_axes([0.8, 0.65, 0.05, 0.2])
    fig.colorbar(im, cax=cax, orientation='vertical')

    ax = fig.add_subplot(gs[1,0])
    im = ax.imshow(object_imag)
    ax.set_title("object_imag")
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


def make_wavefront_sensor_image(N, N_zernike, amplitude_mask, tf_zernike_graph, z_radius, scalef, tf_propagate_gaussian_graph, sess):

    zernike_coefficients = [
            #(m,n)
            # (1,1), # linear phase
            # (-1,1), # linear phase

            (-2,2), # zero at center
            (0,2), # not zero at center
            (2,2), # zero at center

            (-3,3), # zero at center
            (-1,3),
            (1,3),
            (3,3), # zero at center
            (-4,4),
            (-2,4),
            (0,4),
            (2,4),
            (4,4)
            ]

    time1 = time.time()
    zernike_phase = np.zeros((N_zernike,N_zernike))
    for z_coefs in zernike_coefficients:

        # zernike_coef_phase, z_radius = diffraction_functions.zernike_polynomial(N_zernike,z_coefs[0],z_coefs[1], scalef)
        if z_coefs[0] >= 0:
            zernike_coef_phase = sess.run(tf_zernike_graph["Z_even"],
                feed_dict={
                    tf_zernike_graph["m_ph"]:np.abs(z_coefs[0]),
                    tf_zernike_graph["n_ph"]:np.abs(z_coefs[1])
                    })
        else:
            zernike_coef_phase = sess.run(tf_zernike_graph["Z_odd"],
                feed_dict={
                    tf_zernike_graph["m_ph"]:np.abs(z_coefs[0]),
                    tf_zernike_graph["n_ph"]:np.abs(z_coefs[1])
                    })
        zernike_coef_phase*= (-1 + 2*np.random.rand()) # between -1 and 1
        # zernike_coef_phase -= zernike_coef_phase[int(N_zernike/2), int(N_zernike/2)]
        zernike_phase += zernike_coef_phase

    time2 = time.time()
    duration = time2 - time1
    print("generate zernike =>", duration)


    time1 = time.time()
    prop = sess.run(tf_propagate_gaussian_graph["prop"],
            feed_dict={tf_propagate_gaussian_graph["zernike_phase"]:zernike_phase}
            )
    time2 = time.time()
    duration = time2 - time1
    print("propagate =>", duration)

    # fig3 = plot_complex("After FT {0:.5g}".format(scalef), prop, 3, zoom_in=None, axis_limit=100)

    # take the center of propagated beam
    random_shift = ( (-1 + 2*np.random.rand()), (-1 + 2*np.random.rand()) )
    random_shift_scalar = 25
    index_min = int((N_zernike/2) - 100)
    index_max = int((N_zernike/2) + 100)

    prop_center = prop[int(index_min+random_shift_scalar*random_shift[0]):int(index_max+random_shift_scalar*random_shift[0]), int(index_min+random_shift_scalar*random_shift[1]):int(index_max+random_shift_scalar*random_shift[1])]

    # interpolate this onto a N by N grid
    prop_center_N_real = resize(np.real(prop_center), (N,N))
    prop_center_N_imag = resize(np.imag(prop_center), (N,N))
    prop_center_N = prop_center_N_real + 1j * prop_center_N_imag
    prop_center_N *= amplitude_mask
    prop_center_N *=1/(np.max(prop_center_N)) # normalize

    real_masked_prop = np.real(prop_center_N)
    imag_masked_prop = np.imag(prop_center_N)

    return real_masked_prop, imag_masked_prop

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
        hdf5file.create_earray(hdf5file.root, "object_real", tables.Float64Atom(), shape=(0,N*N))

        # create array for the object phase
        hdf5file.create_earray(hdf5file.root, "object_imag", tables.Float64Atom(), shape=(0,N*N))

        # create array for the image
        hdf5file.create_earray(hdf5file.root, "diffraction", tables.Float64Atom(), shape=(0,N*N))

        hdf5file.create_earray(hdf5file.root, "N", tables.Int32Atom(), shape=(0,1))

        hdf5file.close()


    with tables.open_file(filename, mode='a') as hd5file:

        # save the dimmensions of the data
        hd5file.root.N.append(np.array([[N]]))
        # plt.ion()

        N = 128
        # N must be divisible by 4
        assert N/4 == int(N/4)
        # get the png image for amplitude
        im = Image.open("size_6um_pitch_600nm_diameter_300nm_psize_5nm.png")
        im = PIL.ImageOps.invert(im)
        im = im.resize((int(N/2),int(N/2)))
        amplitude_mask = np.array(im.getdata(), dtype=np.uint8).reshape(im.size[0], im.size[1], -1)
        amplitude_mask = np.sum(amplitude_mask, axis=2)
        # pad the amplitude image with zeros
        amplitude_mask = np.concatenate((amplitude_mask, np.zeros((int(N/2),int(N/4)))), axis=1)
        amplitude_mask = np.concatenate((np.zeros((int(N/2),int(N/4))), amplitude_mask), axis=1)
        amplitude_mask = np.concatenate((np.zeros((int(N/4),N)), amplitude_mask), axis=0)
        amplitude_mask = np.concatenate((amplitude_mask, np.zeros((int(N/4),N))), axis=0)
        amplitude_mask *= 1/np.max(amplitude_mask) # normalize
        # amplitude_mask[amplitude_mask>0.5] = 1
        # concat 32
        # prepare tensorflow zernike graph

        N_zernike = 1024
        scalef = 0.6
        tf_zernike_graph, z_radius = build_tf_zernike_graph(N_zernike, scalef)
        tf_propagate_gaussian_graph = build_tf_propagate_gaussian_graph(N_zernike, scalef)
        with tf.Session() as sess:
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

                # plot_thing(object_phase, 1, "object_phase")
                # plot_thing(object_amplitude, 2, "object_amplitude")

                print("====================")
                time1 = time.time()
                object_real, object_imag = make_wavefront_sensor_image(N, N_zernike, amplitude_mask, tf_zernike_graph, z_radius, scalef, tf_propagate_gaussian_graph, sess)
                time2 = time.time()
                duration = time2 - time1
                print("total make_wavefront_sensor_image function duration =>", duration)
                continue
                exit()


                # plot_thing(object_real, 1, "object_real")
                # plot_thing(object_imag, 2, "object_imag")
                complex_object = object_real + 1j*object_imag

                diffraction_pattern = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(complex_object)))
                # absolute value
                diffraction_pattern = np.abs(diffraction_pattern)**2

                gaussian_noise = np.random.normal(scale=0.05*np.max(diffraction_pattern), size=diffraction_pattern.shape)
                diffraction_pattern_with_noise = diffraction_pattern + gaussian_noise

                # normalize the diffraction pattern
                diffraction_pattern_with_noise = diffraction_pattern_with_noise / np.max(diffraction_pattern_with_noise)

                # adjust the real and imaginary parts to be between 0 and 1
                # object_real: # between -1 and 1
                # object_imag: # between -1 and 1
                object_real += 1 # between 0 and 2
                object_imag += 1 # between 0 and 2
                object_real *= (1/2) # between 0 and 1
                object_imag *= (1/2) # between 0 and 1

                if i % 100 == 0:
                    plot_sample(N, object_real, object_imag, diffraction_pattern_with_noise)
                    plt.pause(0.001)

                hd5file.root.object_real.append(object_real.reshape(1,-1))
                hd5file.root.object_imag.append(object_imag.reshape(1,-1))
                # hd5file.root.phase_norm_factor.append(phase_norm_factor.reshape(1,1))
                hd5file.root.diffraction.append(diffraction_pattern_with_noise.reshape(1,-1))

                # # reconstruct phase from label
                # object_phase *= 2 # between 0 and 2
                # object_phase -= 1 # between -1 and 1
                # object_phase *= phase_norm_factor
                # plot_thing(object_phase, 102, "object_phase reconstruct")


                # # reconstruct diffraction pattern
                # recons_diff = diffraction_functions.construct_diffraction_pattern(object_amplitude, object_phase, phase_norm_factor)
                # plt.figure()
                # plt.imshow(recons_diff)
                # plt.colorbar()
                # plt.savefig("./4.png")
                # os.system("display 4.png & disown")
                # exit()


if __name__ == "__main__":
    # generate a data set
    N = 128

    make_dataset("train_data.hdf5", N=N, samples=40000)

    make_dataset("test_data.hdf5", N=N, samples=200)

    # test open the data set
    index = 4
    with tables.open_file("train_data.hdf5", mode="r") as hdf5file:

        # print("hdf5file.root.N =>", hdf5file.root.N[0,0])
        N = hdf5file.root.N[0,0]

        object_real = hdf5file.root.object_real[index,:].reshape(N,N)
        object_imag = hdf5file.root.object_imag[index,:].reshape(N,N)
        diffraction = hdf5file.root.diffraction[index,:].reshape(N,N)

        # plt.figure(2)
        # plt.pcolormesh(object)
        # plt.figure(3)
        # plt.pcolormesh(object_phase)
        # plt.figure(4)
        # plt.pcolormesh(diffraction)
        # plt.show()



import numpy as np
import os
import math
from PIL import Image, ImageDraw
from PIL import ImagePath
import tensorflow as tf
import PIL.ImageOps
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import rotate
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import shift as sc_shift
from  astropy.io import fits
from scipy.misc import factorial
from skimage.transform import resize
from scipy import ndimage

def plot_image_show_centroid_distance(mat, title, figurenum):
    """
    plots an image and shows the distance from the centroid to the image center
    """
    # calculate the current centroid
    c_y = calc_centroid(mat, axis=0)
    c_x = calc_centroid(mat, axis=1)
    plt.figure(figurenum)
    plt.imshow(mat)
    plt.axvline(x=c_x, color="yellow")
    plt.axhline(y=c_y, color="yellow")
    plt.title(title)
    plt.gca().text(0.0, 0.9,"c_x: {}".format( c_x ), fontsize=10, ha='center', transform=plt.gca().transAxes, backgroundcolor="yellow")
    plt.gca().text(0.0, 0.8,"c_y: {}".format( c_y ), fontsize=10, ha='center', transform=plt.gca().transAxes, backgroundcolor="yellow")

    plt.gca().text(0.0, 0.7,"image center x: {}".format( np.shape(mat)[0] / 2 ), fontsize=10, ha='center', transform=plt.gca().transAxes, backgroundcolor="green")
    plt.gca().text(0.0, 0.6,"image center y: {}".format( np.shape(mat)[0] / 2 ), fontsize=10, ha='center', transform=plt.gca().transAxes, backgroundcolor="green")

    # center of the image
    plt.axvline(x=int(np.shape(mat)[0] / 2), color="green")
    plt.axhline(y=int(np.shape(mat)[0] / 2), color="green")

def format_experimental_trace(N, df_ratio, measured_diffraction_pattern, rotation_angle):
    """
    N: the desired size of the formatted experimental image
    df_ratio: (df calculated from diffraction plane) / (df calculated from object plane)
    measured_diffraction_pattern: the measured diffraction pattern to format
    rotation_angle: angle which to rotate the measured diffraction pattern (currently must be done by eye)
    """
    # divide scale the measured trace by this amount
    new_size = np.shape(measured_diffraction_pattern)[0] * df_ratio
    new_size = int(new_size)
    measured_diffraction_pattern = resize(measured_diffraction_pattern, (new_size, new_size))

    # rotate the image by eye
    measured_diffraction_pattern = ndimage.rotate(measured_diffraction_pattern, rotation_angle, reshape=False)
    measured_diffraction_pattern = center_image_at_centroid(measured_diffraction_pattern)
    # crop the edges off the image
    measured_diffraction_pattern = measured_diffraction_pattern[int((new_size/2) - (N/2)):int((new_size/2) + (N/2)) , int((new_size/2) - (N/2)):int((new_size/2) + (N/2))]
    measured_diffraction_pattern = center_image_at_centroid(measured_diffraction_pattern)

    return measured_diffraction_pattern

def center_image_at_centroid(mat):

    s_x, s_y = np.shape(mat)
    assert s_y == s_x

    c_y = calc_centroid(mat, axis=0)
    c_x = calc_centroid(mat, axis=1)
    # distance from centroid
    dis_x = (s_x / 2) - c_x
    dis_y = (s_x / 2) - c_y
    mat = np.roll(mat, shift=int(dis_x), axis=1)
    mat = np.roll(mat, shift=int(dis_y), axis=0)
    return mat


def get_measured_diffraction_pattern_grid():
    """
    measured_pattern: (numpy array)

    experimental_params: (dict)
    experimental_params['pixel_size'] (meters)
    experimental_params['z_distance'] (meters)
    experimental_params['wavelength'] (meters)

    """

    fits_file_name = "m3_scan_0000.fits"
    thing = fits.open(fits_file_name)
    measured_pattern = thing[0].data[0,:,:]

    experimental_params = {}
    experimental_params['pixel_size'] = 27e-6 # [meters] with 2x2 binning
    experimental_params['z_distance'] = 33e-3 # [meters] distance from camera
    experimental_params['wavelength'] = 13.5e-9 #[meters] wavelength


    assert np.shape(measured_pattern)[0] == np.shape(measured_pattern)[1]
    # construct position (space) axis
    # print("np.shape(measured_pattern) => ",np.shape(measured_pattern))
    N = np.shape(measured_pattern)[0]

    # calculate delta frequency
    measured_axes = {}
    measured_axes["diffraction_plane"] = {}

    measured_axes["diffraction_plane"]["xmax"] = N * (experimental_params['pixel_size'] / 2)
    measured_axes["diffraction_plane"]["x"] = np.arange(-(measured_axes["diffraction_plane"]["xmax"]), (measured_axes["diffraction_plane"]["xmax"]), experimental_params['pixel_size'])

    measured_axes["diffraction_plane"]["f"] = measured_axes["diffraction_plane"]["x"] / (experimental_params['wavelength'] * experimental_params['z_distance'])
    measured_axes["diffraction_plane"]["df"] = measured_axes["diffraction_plane"]["f"][-1] - measured_axes["diffraction_plane"]["f"][-2]

    return measured_axes, measured_pattern

def get_amplitude_mask_and_imagesize(image_dimmension, desired_mask_width):

        # image_dimmension must be divisible by 4
        assert image_dimmension/4 == int(image_dimmension/4)
        # get the png image for amplitude
        im = Image.open("size_6um_pitch_600nm_diameter_300nm_psize_5nm.png")
        im = PIL.ImageOps.invert(im)

        size1 = im.size[0]
        im_size_nm = 5*im.size[0] * 1e-9 # meters
        # print("im_size_nm =>", im_size_nm)

        # scale down the image
        im = im.resize((desired_mask_width,desired_mask_width)).convert("L")
        size2 = im.size[0]
        im = np.array(im)

        # # # # # # # # # # # # #
        # # # pad the image # # #
        # # # # # # # # # # # # #

        # determine width of mask
        pad_amount = int((image_dimmension - desired_mask_width)/2)
        amplitude_mask = np.pad(im, pad_width=pad_amount, mode="constant", constant_values=0)
        amplitude_mask = amplitude_mask.astype(np.float64)
        amplitude_mask *= 1/np.max(amplitude_mask) # normalize
        assert amplitude_mask.shape[0] == image_dimmension
        size3 = np.shape(amplitude_mask)[0]
        ratio = size3 / size2
        # print("ratio =>", ratio)
        im_size_nm *= ratio # object image size [m]

        measured_axes = {}
        measured_axes["object"] = {}
        measured_axes["object"]["dx"] = im_size_nm / image_dimmension
        measured_axes["object"]["xmax"] = im_size_nm/2
        measured_axes["object"]["x"] = np.arange(-(measured_axes["object"]["xmax"]), (measured_axes["object"]["xmax"]), measured_axes["object"]["dx"])
        measured_axes["diffraction_plane"] = {}
        measured_axes["diffraction_plane"]["df"] = 1 / (image_dimmension * measured_axes["object"]["dx"]) # frequency axis in diffraction plane
        measured_axes["diffraction_plane"]["fmax"] = (measured_axes["diffraction_plane"]["df"] * image_dimmension) / 2
        measured_axes["diffraction_plane"]["f"] = np.arange(-measured_axes["diffraction_plane"]["fmax"], measured_axes["diffraction_plane"]["fmax"], measured_axes["diffraction_plane"]["df"])

        return measured_axes, amplitude_mask

def make_image_square(image):

    # make the image square
    rows = np.shape(image)[0]
    cols = np.shape(image)[1]
    if rows > cols:
        half_new_rows = int(cols / 2)
        center_row = int(rows / 2)
        im_new = image[center_row-half_new_rows:center_row+half_new_rows,:]
    if cols > rows:
        half_new_cols = int(rows / 2)
        center_col = int(cols / 2)
        im_new = image[:,center_col-half_new_cols:center_col+half_new_cols]

    return im_new

def bin_image(image, bin_shape):
    im_shape = image.shape
    image = image.reshape(
            int(im_shape[0]/bin_shape[0]),
            bin_shape[0],
            int(im_shape[1]/bin_shape[1]),
            bin_shape[1]
            ).sum(3).sum(1)
    return image


def circular_crop(image, radius):

    # multpily by a circular beam amplitude
    y = np.linspace(-1, 1, np.shape(image)[0]).reshape(-1,1)
    x = np.linspace(-1, 1, np.shape(image)[1]).reshape(1,-1)
    r = np.sqrt(x**2 + y**2)
    # multiply the object by the beam
    image[r>radius] = 0


def rescale_image(image, scale):
    new_img_size = (np.array(image.size) * scale).astype(int)  # (width, height)
    c_left = (image.size[0] / 2) - (new_img_size[0]/2)
    c_upper = (image.size[1] / 2) - (new_img_size[1]/2)
    c_right = (image.size[0] / 2) + (new_img_size[0]/2)
    c_lower = (image.size[1] / 2) + (new_img_size[1]/2)
    image = image.crop((c_left,c_upper,c_right,c_lower))# left , upper, right, lower
    return image


def zernike_polynomial(N, m, n, scalef):

    if m >= 0:
        even = True
    else:
        even = False
    n = np.abs(n)
    m = np.abs(m)

    # assert (n-m)/2 is an integer
    assert float((n-m)/2) - int((n-m)/2) == 0

    # make axes of rho and phi
    scale = 10.0/scalef
    x = np.linspace(-1*scale,1*scale,N).reshape(1,-1)
    y = np.linspace(-1*scale,1*scale,N).reshape(-1,1)

    rho = np.sqrt(x**2 + y**2)
    rho = np.expand_dims(rho, axis=2)

    phi = np.arctan2(y,x)

    # define axes
    # rho = np.linspace(0, 1, 100).reshape(1,-1)
    k = np.arange(0, ((n-m)/2)+1 ).reshape(1,1,-1)

    numerator = (-1)**k
    numerator *= factorial(n-k)

    denominator = factorial(k)
    denominator *= factorial(((n+m)/2)-k)
    denominator *= factorial(((n-m)/2)-k)

    R = (numerator / denominator)*rho**(n-2*k)
    R = np.sum(R, axis=2)

    if even:
        Z = R*np.cos(m*phi)
    else:
        Z = R*np.sin(m*phi)

    # for checking the sampling
    # the -1 <-> 1 range of the zernike polynomial should be approximately the width of the
    # not propagated pulse
    # use this to set scale
    r = np.sqrt(x**2 + y**2)
    # Z[r>1] = 0


    return Z, r



def tf_reconstruct_diffraction_pattern(real_norm, imag_norm):

    real_norm *= 2 # between 0 and 2
    imag_norm *= 2 # between 0 and 2

    real_norm -= 1 # between -1 and 1
    imag_norm -= 1 # between -1 and 1

    complex_object_retrieved = tf.complex(real=real_norm, imag=imag_norm)
    diffraction_pattern = tf.abs(tf_fft2(complex_object_retrieved, dimmensions=[1,2]))**2

    diffraction_pattern = diffraction_pattern / tf.reduce_max(diffraction_pattern, keepdims=True, axis=[1,2]) # normalize the diffraction pattern
    return diffraction_pattern

def construct_diffraction_pattern(normalized_amplitude, normalized_phase, scalar):
    """
    construct diffraction pattern from normalized (retrieved object)

    """
    amplitude = np.array(normalized_amplitude)
    phase = np.array(normalized_phase)

    phase *= scalar

    complex_object = amplitude * np.exp(1j * phase)

    diffraction_pattern = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(complex_object)))
    # absolute value
    diffraction_pattern = np.abs(diffraction_pattern)
    # normalize the diffraction pattern
    diffraction_pattern = diffraction_pattern / np.max(diffraction_pattern)

    return diffraction_pattern

def tf_fft2(image_in, dimmensions):
    """
        2D fourer transform matrix along dimmensions

        image_in: n-dimmensional complex tensor
        dimmensions: the dimmensions to do 2D FFt
    """
    assert len(dimmensions) == 2

    # image_shifted = np.array(image_in)
    for _i in dimmensions:
        assert int(image_in.shape[_i]) % 2 == 0
        dim_shift = int(int(image_in.shape[_i]) / 2)
        image_in = tf.manip.roll(image_in, shift=dim_shift, axis=_i)

    # function is only made for inner two dimmensions to be fourier transformed
    # assert image_in.shape[0] == 1
    assert image_in.shape[3] == 1

    image_in = tf.transpose(image_in, perm=[0,3,1,2])
    image_in = tf.fft2d(image_in)
    image_in = tf.transpose(image_in, perm=[0,2,3,1])

    for _i in dimmensions:
        dim_shift = int(int(image_in.shape[_i]) / 2)
        image_in = tf.manip.roll(image_in, shift=dim_shift, axis=_i)

    return image_in


def f_position_shift(mat, shift_value, axis):
    # shift_value = 1.5
    """
    mat: 2d numpy array
    shift_value: the number of columns/rows to shift the matrix
    axis: the axis to shift the position

    shift_value may be a float

    """
    # print("shift_value =>", shift_value)

    shift_val = [0,0]
    shift_val[axis] = shift_value
    mat = sc_shift(mat, shift=tuple(shift_val))

    return mat

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
    # cast value to integer
    value = int(value)

    # print("value =>", value)

    # calculate current centroid:
    start_c = calc_centroid(np.abs(mat), axis)
    target_c = start_c + value
    # print("start_c =>", start_c)
    delta_c = 0

    if value > 0:
        # increase the centroid while its less than the target
        new_c = float(start_c)
        while new_c < target_c:
            mat = np.roll(mat, shift=1, axis=axis)
            new_c = calc_centroid(np.abs(mat), axis)

    elif value < 0:
        # decrease the centroid while its greater than the target
        new_c = float(start_c)

        while new_c > target_c:
            mat = np.roll(mat, shift=-1, axis=axis)
            new_c = calc_centroid(np.abs(mat), axis)

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
    object: 2d numpy array (not complex)

    remove the translation and conjugate flip ambiguities
    of a 2d complex matrix
    """

    obj_size = np.shape(object)
    target_row = int(obj_size[0]/2)
    target_col = int(obj_size[1]/2)

    # calculate centroid along rows
    centr_row = calc_centroid(object, axis=0)
    centr_col = calc_centroid(object, axis=1)

    # move centroid to the center
    object = f_position_shift(object, shift_value=(target_row-centr_row), axis=0)
    object = f_position_shift(object, shift_value=(target_col-centr_col), axis=1)
    # remove conjugate flip ambiguity

    # integrate upper left and bottom right triangle
    # lower left
    tri_l = np.tril(np.ones(np.shape(object)))
    # upper right
    tri_u = np.triu(np.ones(np.shape(object)))
    integral_upper = np.sum(tri_u*object, axis=(0,1))
    integral_lower = np.sum(tri_l*object, axis=(0,1))

    # print(integral_upper > integral_lower)
    if integral_upper > integral_lower:
        # make conjugate flip
        object = np.flip(object, axis=1)
        object = np.flip(object, axis=0)

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
    # must be divisible by 4
    assert N % 4  == 0
    obj = np.zeros((N,N), dtype=np.complex128)

    min_x = N/4 + 1
    min_y = N/4 + 1
    max_x = N - N/4 - 1
    max_y = N - N/4 - 1

    # generate random indexes
    # np.random.seed(3367)
    indexes_n = np.random.randint(min_indexes,max_indexes)
    # for each index generate an x and y point
    x = []
    y = []
    for i in range(indexes_n):

        x_val = min_x + np.random.rand(1)*(max_x-min_x)
        y_val = min_y + np.random.rand(1)*(max_y-min_y)

        x.append(int(x_val))
        y.append(int(y_val))

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
    return amplitude

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

    return fig

def create_phase(N):
    """
    N: dimmensions of image

    returns:
    phase from -pi to +pi

    the phase is 0 at the center of the image

    """
    # np.random.seed(22)
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

    phase = np.exp(1j * phase_frequency * x_rot) * np.exp(1j * 10*np.random.rand())

    # subtract phase at center
    # phase_at_center = np.angle(phase[int(N/2), int(N/2)])
    # phase = phase * np.exp(-1j * phase_at_center) * np.exp(1j * np.pi)
    phase = np.pi*phase

    # from - pi to + pi
    return np.real(phase)


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


if __name__ == "__main__":

    # construct object in the object plane
    object, object_phase = make_object(N, min_indexes=4, max_indexes=8)
    # object_phase = np.ones_like(object_phase)
    # construct phase
    object_with_phase = make_object_phase(object, object_phase)

    object_with_phase_removed_ambi = remove_ambiguitues(np.array(object_with_phase))

    plt.figure(1)
    plt.pcolormesh(np.abs(object_with_phase))
    plt.figure(2)
    plt.pcolormesh(np.angle(object_with_phase))

    plt.figure(3)
    plt.pcolormesh(np.abs(object_with_phase_removed_ambi))
    plt.figure(4)
    plt.pcolormesh(np.angle(object_with_phase_removed_ambi))

    plt.show()
    exit()


    fig = plot_fft(object_with_phase)
    fig = plot_fft(object_with_phase_removed_ambi)
    # fig.savefig("obj1.png")

    # apply roll to generate ambiguity
    # fig = plot_fft(make_roll_ambiguity(object_with_phase))
    # fig.savefig("obj2.png")

    # # conjugate flip to generate ambiguity
    # fig = plot_fft(make_flip_ambiguity(object_with_phase))
    # fig.savefig("obj3.png")

    plt.show()








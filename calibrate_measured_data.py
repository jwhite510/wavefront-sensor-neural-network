import numpy as np
from  astropy.io import fits
import tables
import matplotlib.pyplot as plt
import pickle
from scipy import signal
from PIL import Image
import PIL.ImageOps
from generate_data import plot_complex
from scipy import signal
import cv2
import time


def find_maxima(pattern, number_of_maxima):

    # peak_value = np.argmax(pattern)
    peak_value = np.unravel_index(np.argmax(pattern, axis=None), pattern.shape)
    print("peak_value =>", peak_value)


def compare_autoc_and_fft(complex_object):
    # make autocorrelation of complex_object
    complex_object_auto = signal.correlate2d(complex_object, complex_object, mode="same")
    # exit()

    e_w = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(complex_object)))
    I = np.abs(e_w)**2
    I_fft = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(I)))

    plt.figure()
    plt.imshow(np.abs(complex_object_auto))
    plt.title("np.abs(complex_object_auto)")
    plt.figure()
    plt.imshow(np.real(complex_object_auto))
    plt.title("np.real(complex_object_auto)")
    plt.figure()
    plt.imshow(np.imag(complex_object_auto))
    plt.title("np.imag(complex_object_auto)")

    plt.figure()
    plt.imshow(np.abs(I_fft))
    plt.title("np.abs(I_fft)")
    plt.figure()
    plt.imshow(np.real(I_fft))
    plt.title("np.real(I_fft)")
    plt.figure()
    plt.imshow(np.imag(I_fft))
    plt.title("np.imag(I_fft)")



def compare_autoc_meas_sim(simulated_diffraction, measured_diffraction):

    plt.figure()
    plt.title("measured data")
    plt.imshow(measured_diffraction)

    # simulated data
    plt.figure()
    plt.imshow(simulated_diffraction)
    plt.title("diffraction simulated")


    # fourier transform of simulated diffraction pattern
    simulated_autoc = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(simulated_diffraction)))
    simulated_autoc = np.abs(simulated_autoc)
    plt.figure()
    plt.imshow(simulated_autoc)
    plt.title("simulated_autoc")

    measured_autoc = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(measured_diffraction)))
    measured_autoc = np.abs(measured_autoc)
    plt.figure()
    plt.imshow(measured_autoc)
    plt.title("measured_autoc")


    plt.show()

def make_autocorrelation(Intensity):
    return np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(Intensity)))

def find_center_of_maxima(image):
    assert np.shape(image)[0] == np.shape(image)[1]

    print("np.shape(image) => ",np.shape(image))
    plt.figure()
    plt.imshow(image)
    plt.title("image")

    dilatation_size = 9
    # dilatation_type = cv2.MORPH_RECT
    # dilatation_type = cv2.MORPH_CROSS
    dilatation_type = cv2.MORPH_ELLIPSE
    element = cv2.getStructuringElement(2, (2*dilatation_size + 1, 2*dilatation_size+1), (dilatation_size, dilatation_size))

    measured_dilated = cv2.dilate(image, element)
    # measured_eroded = cv2.dilate(measured_dilated, element)
    # measured_eroded = cv2.dilate((image - measured_dilated), element)

    # convert to uint8
    measured_dilated *= 1/(np.max(measured_dilated))
    measured_dilated *= 255
    measured_dilated = measured_dilated.astype(np.uint8)

    plt.figure()
    plt.imshow(np.array(measured_dilated))
    plt.colorbar()
    plt.title("measured_dilated as uint8, converted")

    measured_dilated = cv2.medianBlur(measured_dilated,5)
    cimg = cv2.cvtColor(measured_dilated,cv2.COLOR_GRAY2BGR)
    circles = cv2.HoughCircles(measured_dilated,cv2.HOUGH_GRADIENT,1,20,
                                param1=50,param2=30,minRadius=0,maxRadius=0)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            # draw the outer circle
            cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

    # cv2.imshow('detected circles',cimg)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    plt.figure()
    plt.imshow(cimg)
    plt.title("cimg")


def align_image(measured, image_from_dataset):

    find_center_of_maxima(measured)
    plt.show()
    find_center_of_maxima(image_from_dataset)
    plt.show()
    exit()

if __name__ == "__main__":

    # open hdf5 samples

    index = 4
    with tables.open_file("train_data.hdf5", mode="r") as hdf5file:

        N = hdf5file.root.N[0,0]
        object_real = hdf5file.root.object_real[index,:].reshape(N,N)
        object_imag = hdf5file.root.object_imag[index,:].reshape(N,N)
        diffraction = hdf5file.root.diffraction[index,:].reshape(N,N)
    object_real *= 2
    object_imag *= 2
    object_real -= 1
    object_imag -= 1
    complex_object = object_real + 1j * object_imag

    # measured data
    fits_file_name = "m3_scan_0000.fits"
    thing = fits.open(fits_file_name)
    measured_diffraction_pattern = thing[0].data[0,:,:]

    # print("measured_diffraction_pattern.dtype =>", measured_diffraction_pattern.dtype)
    # exit()

    # im = Image.fromarray(measured_diffraction_pattern).convert("L")
    # im = np.array(im)

    measured_diffraction_pattern = measured_diffraction_pattern.astype(np.float64)

    align_image(measured_diffraction_pattern, diffraction)
    exit()

    plt.figure()
    plt.title("diffraction")
    plt.imshow(diffraction)

    plt.figure()
    plt.title("np.abs(make_autocorrelation(diffraction))")
    plt.imshow(np.abs(make_autocorrelation(diffraction)))

    plt.figure()
    plt.title("measured_diffraction_pattern")
    plt.imshow(measured_diffraction_pattern)

    plt.figure()
    plt.title("np.abs(make_autocorrelation(measured_diffraction_pattern))")
    plt.imshow(np.abs(make_autocorrelation(measured_diffraction_pattern)))

    plt.show()
    exit()

    plt.figure()
    plt.imshow(measured_diffraction_pattern)
    plt.colorbar()
    plt.savefig("measured_diffraction_pattern.png")

    im = Image.open("size_6um_pitch_600nm_diameter_300nm_psize_5nm.png")
    im = PIL.ImageOps.invert(im)
    image_width = int(N/4)
    im = im.resize((image_width,image_width)).convert("L")
    im = np.array(im)
    im[im > 0.01] = 1

    # determine width of mask
    pad_amount = int((N - image_width)/2)
    amplitude_mask = np.pad(im, pad_width=pad_amount, mode="constant", constant_values=0)
    amplitude_mask = amplitude_mask.astype(np.float64)
    amplitude_mask *= 1/np.max(amplitude_mask) # normalize
    assert amplitude_mask.shape[0] == N

    # apply phase
    range_val = 10
    x = np.linspace(-range_val, range_val, N).reshape(-1,1 )
    y = np.linspace(-range_val, range_val, N).reshape(1,-1)
    phi = x**2 * y**2
    plt.figure()
    plt.imshow(phi)
    plt.colorbar()
    plt.savefig("phi.png")
    w = 1.5
    gaussian = np.exp((-x**2) / w**2) * np.exp((-y**2) / w**2)

    simulated_object = gaussian * amplitude_mask * np.exp(1j * phi)

    f = plot_complex("simulated_object", simulated_object, 88)
    f.savefig("simulated_object.png")


    diffraction_pattern = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(simulated_object)))
    # absolute value
    diffraction_pattern = np.abs(diffraction_pattern)**2

    f = plot_complex("diffraction_pattern", diffraction_pattern, 89)
    f.savefig("diffraction_pattern.png")
    plt.show()


    # plt.figure()
    # plt.imshow(np.abs(complex_object))
    # plt.title("np.abs(complex_object)")

    # plt.show()
    # compare_autoc_meas_sim(diffraction, thing[0].data[0,:,:])
    # compare_autoc_meas_sim(diffraction, thing[0].data[0,:,:])


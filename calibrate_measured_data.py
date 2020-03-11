import numpy as np
from  astropy.io import fits
import tables
import matplotlib.pyplot as plt
from scipy import signal


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



def compare_autoc_meas_sim(simulated_diffraction, measured_diffraction, complex_object):

    plt.figure()
    plt.title("measured data")
    plt.imshow(thing[0].data[0,:,:])

    # simulated data
    plt.figure()
    plt.imshow(diffraction)
    plt.title("diffraction simulated")

    plt.figure()
    plt.imshow(np.abs(complex_object))
    plt.title("np.abs(complex_object)")

    plt.figure()
    plt.imshow(np.angle(complex_object))
    plt.title("np.angle(complex_object)")

    # fourier transform of simulated diffraction pattern
    simulated_autoc = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(diffraction)))
    simulated_autoc = np.abs(simulated_autoc)
    plt.figure()
    plt.imshow(simulated_autoc)
    plt.title("simulated_autoc")

    measured_autoc = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(thing[0].data[0,:,:])))
    measured_autoc = np.abs(measured_autoc)
    plt.figure()
    plt.imshow(measured_autoc)
    plt.title("measured_autoc")


    plt.show()

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
    fits_file_name = "/home/jonathon/Documents/test/windowshare/1.fits"
    thing = fits.open(fits_file_name)

    compare_autoc_meas_sim(diffraction, thing[0].data[0,:,:], complex_object)





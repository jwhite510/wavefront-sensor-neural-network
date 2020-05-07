import numpy as np
import diffraction_functions
import matplotlib.pyplot as plt

def get_object_sensor(a):
    N = 1024
    _, b = diffraction_functions.get_amplitude_mask_and_imagesize2(N, int(N/2))
    b = b.astype(np.float32)
    return b
    # return np.ones((128,128)).astype(np.float32)

def get_object_sensor_f(a):
    N = 1024
    measured_axes, b = diffraction_functions.get_amplitude_mask_and_imagesize2(N, int(N/2))
    # b = b.astype(np.float32)
    b = measured_axes["diffraction_plane"]["f"].astype(np.float32)
    return b
    # return np.ones((128)).astype(np.float32)

def plot(a):
    print("np.shape(a) => ",np.shape(a))
    plt.figure()
    plt.imshow(a)
    plt.show()

def plot_complex(array):
    fig = plt.figure(figsize=(10,4))
    gs = fig.add_gridspec(1,4)

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

    ax = fig.add_subplot(gs[0,3])
    im = ax.imshow(np.abs(array))
    fig.colorbar(im, ax=ax)
    ax.set_title("abs")

def show(a):
    plt.show()



if __name__ == "__main__":
    get_object_sensor(None)


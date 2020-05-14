import numpy as np
import diffraction_functions
import matplotlib.pyplot as plt

def get_object_sensor(a):
    N = 1024
    _, b = diffraction_functions.get_amplitude_mask_and_imagesize2(N, int(N/2))
    # _, b = diffraction_functions.get_amplitude_mask_and_imagesize(N, int(N/2))
    b = b.astype(np.float32)
    return b
    # return np.ones((128,128)).astype(np.float32)

def get_object_sensor_f(a):
    N = 1024
    measured_axes, b = diffraction_functions.get_amplitude_mask_and_imagesize2(N, int(N/2))
    # measured_axes, b = diffraction_functions.get_amplitude_mask_and_imagesize(N, int(N/2))
    # b = b.astype(np.float32)
    b = measured_axes["diffraction_plane"]["f"].astype(np.float32)
    return b
    # return np.ones((128)).astype(np.float32)

def plot(a):
    print("np.shape(a) => ",np.shape(a))
    plt.figure()
    plt.imshow(a)

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

def plot_large(array):

    plt.figure()
    plt.pcolormesh(np.angle(array),cmap='jet')
    plt.title("angle")
    plt.gcf().text(0.5, 0.95, "my propagation code", ha="center")
    plt.colorbar()

def show(a):
    plt.show()

class Parameters():
    def __init__(self):
        self.beta_Ta=None
        self.delta_Ta=None
        self.dz=10e-9
        self.lam=18.1e-9
        self.k=2*np.pi/self.lam

def create_slice(slice, object, p):
    for i in range(0,slice.shape[0]):
        for j in range(0,slice.shape[1]):
            if object[i,j]<0.5:
                slice[i,j] =np.exp(-1 *p.k*p.beta_Ta*p.dz)
                slice[i,j]*=np.exp(-1j*p.k*p.delta_Ta*p.dz)
            else:
                slice[i,j]=1

def forward_propagate(E, slice, f, p):

    E*=slice

    E=np.fft.fftshift(np.fft.fft2(np.fft.fftshift(E)))
    gamma=np.sqrt(1-(p.lam*f.reshape(-1,1))**2 - (p.lam*f.reshape(1,-1))**2)
    k_sq = 2 * np.pi * p.dz / p.lam;
    H = np.exp(1j*gamma*k_sq)
    E*=H
    E=np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(E)))

    return E



if __name__ == "__main__":
    object = get_object_sensor(None)
    f = get_object_sensor_f(None)
    # print("np.shape(object) => ",np.shape(object))
    # plt.imshow(object)
    # plt.show()
    params_cu = Parameters()
    params_cu.beta_Ta =  0.0646711215;
    params_cu.delta_Ta = 0.103724159;
    cu_distance=100e-9
    slice_cu = np.zeros((1024,1024), dtype=np.complex64)
    create_slice(slice_cu, object, params_cu)

    steps=cu_distance/params_cu.dz
    steps=int(steps)

    wave=np.ones((1024,1024),dtype=np.complex64)

    for _ in range(steps):
        wave=forward_propagate(wave, slice_cu, f, params_cu)

    plot_complex(wave)
    plt.show()


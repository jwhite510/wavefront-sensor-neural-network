import pickle
import numpy as np
import matplotlib.pyplot as plt
import utility
import math
import time


if __name__ == "__main__":
    with open("f.p", "rb") as file:
        f = pickle.load(file)

    with open("interped_arr.p", "rb") as file:
        interped_arr = pickle.load(file)

    with open("slice_cu.p", "rb") as file:
        slice_cu = pickle.load(file)

    with open("slice_Si.p", "rb") as file:
        slice_Si = pickle.load(file)

    dz = 1e-9
    lam = 633e-9
    fx_grid = f.reshape(1,-1)
    fy_grid = f.reshape(-1,1)

    # forward propagate
    # print(interped_arr.dtype)

    Si_distance = 50e-9;
    cu_distance = 150e-9;

    steps_Si = int(math.ceil(Si_distance / dz));
    steps_cu = int(math.ceil(cu_distance / dz));

    print("steps_cu =>", steps_cu)
    print("steps_Si =>", steps_Si)

    time1 = time.time()
    for _ in range(steps_Si):
        interped_arr *= slice_Si
        interped_arr = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(interped_arr))).astype(np.complex64)
        H = np.exp(1j * 2 * np.pi * dz / lam * np.sqrt(1 - (lam * fx_grid) ** 2 - (lam * fy_grid) ** 2)).astype(np.complex64)
        interped_arr *= H
        interped_arr = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(interped_arr))).astype(np.complex64)

    for _ in range(steps_cu):
        interped_arr *= slice_cu
        interped_arr = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(interped_arr))).astype(np.complex64)
        H = np.exp(1j * 2 * np.pi * dz / lam * np.sqrt(1 - (lam * fx_grid) ** 2 - (lam * fy_grid) ** 2)).astype(np.complex64)
        interped_arr *= H
        interped_arr = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(interped_arr))).astype(np.complex64)

    # time.sleep(1)
    time2 = time.time()
    duration = time2 - time1
    duration *= 1e6
    print("duration:"+str(duration))

    utility.plot_complex_diffraction(interped_arr)
    plt.show()


import numpy as np
import matplotlib.pyplot as plt


def spatial_grid(x_max, dx, y_max, dy):

    x_grid = np.arange(-x_max, x_max, dx)
    y_grid = np.arange(-y_max, y_max, dy)

    return x_grid, y_grid



if __name__ == "__main__":

    # surface coordinates before propagation
    sigma, eta = spatial_grid(x_max=1e-2, dx=1e-3, y_max=1e-2, dy=1e-3)
    sigma = sigma.reshape(1, -1, 1, 1)
    eta = eta.reshape(-1, 1, 1, 1)

    # define initial field
    U_initial = np.zeros_like(sigma * eta, dtype=np.complex128)
    U_initial[:,8:11] = 1

    plt.figure(1)
    plt.pcolormesh(np.squeeze(sigma), np.squeeze(eta), np.squeeze(np.abs(U_initial)**2))
    plt.show()



    print("np.shape(U_initial) =>", np.shape(U_initial))
    exit()

    # surface coordinates after propagation
    x, y = spatial_grid(x_max=1e-2, dx=1e-3, y_max=1e-2, dy=1e-3)
    x = x.reshape(1, 1, 1, -1)
    y = y.reshape(1, 1, -1, 1)

    z = 1 # meter
    k = 1 # propagation constant for the wavelength
    wavelength = 500*1e-9

    r_01 = np.sqrt(z**2 + (x - sigma)**2 + (y - eta)**2)

    U_initial * np.exp(1j * k * r_01) / (r_01**2)






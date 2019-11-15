import numpy as np
import matplotlib.pyplot as plt


def spatial_grid(x_max, dx, y_max, dy):

    x_grid = np.arange(-x_max, x_max, dx)
    y_grid = np.arange(-y_max, y_max, dy)

    return x_grid, y_grid



if __name__ == "__main__":

    # surface coordinates before propagation
    ds = 1e-3
    sigma, eta = spatial_grid(x_max=1e-2, dx=ds, y_max=1e-2, dy=ds)
    sigma = sigma.reshape(1, -1, 1, 1)
    eta = eta.reshape(-1, 1, 1, 1)

    # define initial field
    U_initial = np.zeros_like(sigma * eta, dtype=np.complex128)
    U_initial[:,8:11] = 1

    plt.figure(1)
    plt.title("U_initial")
    plt.pcolormesh(np.squeeze(sigma), np.squeeze(eta), np.squeeze(np.abs(U_initial)**2))

    # surface coordinates after propagation
    x, y = spatial_grid(x_max=1e-2, dx=ds, y_max=1e-2, dy=ds)
    x = x.reshape(1, 1, 1, -1)
    y = y.reshape(1, 1, -1, 1)

    z = 1 # meter
    k = 1 # propagation constant for the wavelength
    wavelength = 500*1e-9

    r_01 = np.sqrt(z**2 + (x - sigma)**2 + (y - eta)**2)

    fres = U_initial * np.exp(1j * k * r_01) / (r_01**2)

    fres = np.sum(fres, axis=0)*ds
    fres = np.sum(fres, axis=0)*ds

    U_propagated = ( z / (1j * wavelength) ) * fres

    plt.figure(2)
    plt.title("U_propagated")
    plt.pcolormesh(np.squeeze(x), np.squeeze(y), np.squeeze(np.abs(U_propagated)**2))
    print("np.shape(fres) =>", np.shape(fres))

    plt.show()


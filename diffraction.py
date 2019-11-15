import numpy as np
import matplotlib.pyplot as plt


def spatial_grid(x_max, dx, y_max, dy):

    x_grid = np.arange(-x_max, x_max, dx)
    y_grid = np.arange(-y_max, y_max, dy)

    x_grid = x_grid.reshape(1, -1)
    y_grid = y_grid.reshape(-1, 1)

    return x_grid, y_grid





# x = np.linspace(0,10,20)
# y = np.sin(x)

# plt.figure()
# plt.plot(x, y)
# plt.show()



if __name__ == "__main__":

    # surface coordinates before propagation
    sima, eta = spatial_grid(x_max=1e-2, dx=1e-3, y_max=1e-2, dy=1e-3)

    # surface coordinates after propagation
    x, y = spatial_grid(x_max=1e-2, dx=1e-3, y_max=1e-2, dy=1e-3)

    z = 1 # meter
    wavelength = 500*1e-9






import numpy as np
import matplotlib.pyplot as plt


class DiffractionObject():
    def __init__(self, ds, y_max, x_max, wavelength):
        # surface coordinates before propagation
        self.ds = ds
        self.y_max = y_max
        self.wavelength = wavelength
        self.x_max = x_max
        # self.ds = 0.4e-3
        self.sigma, self.eta = spatial_grid(x_max=self.x_max, dx=self.ds, y_max=self.y_max, dy=self.ds)
        self.sigma = self.sigma.reshape(1, -1, 1, 1)
        self.eta = self.eta.reshape(-1, 1, 1, 1)

        # surface coordinates after propagation
        self.x, self.y = spatial_grid(x_max=self.x_max, dx=self.ds, y_max=self.y_max, dy=self.ds)
        self.x = self.x.reshape(1, 1, 1, -1)
        self.y = self.y.reshape(1, 1, -1, 1)

        self.U_initial = None



def spatial_grid(x_max, dx, y_max, dy):

    x_grid = np.arange(-x_max, x_max, dx)
    y_grid = np.arange(-y_max, y_max, dy)

    return x_grid, y_grid

def fresnel_diffraction():

    # surface coordinates before propagation
    ds = 0.4e-3
    sigma, eta = spatial_grid(x_max=1e-2, dx=ds, y_max=1e-2, dy=ds)
    sigma = sigma.reshape(1, -1, 1, 1)
    eta = eta.reshape(-1, 1, 1, 1)

    # define initial field
    U_initial = np.zeros_like(sigma * eta, dtype=np.complex128)
    U_initial[:,18:20] = 1
    U_initial[:,30:32] = 1

    plt.figure(1)
    plt.title("U_initial")
    plt.pcolormesh(np.squeeze(sigma), np.squeeze(eta), np.squeeze(np.abs(U_initial)**2))
    # plt.show()
    # exit()

    # surface coordinates after propagation
    x, y = spatial_grid(x_max=1e-2, dx=ds, y_max=1e-2, dy=ds)
    x = x.reshape(1, 1, 1, -1)
    y = y.reshape(1, 1, -1, 1)

    # k = 1 # propagation constant for the wavelength
    wavelength = 500*1e-9
    k = 2*np.pi / wavelength


    z = 1 # meter
    z = np.linspace(1000*wavelength,1003*wavelength,100)
    plt.ion()
    fig, ax = plt.subplots()
    for z_ in z:

        r_01 = np.sqrt(z_**2 + (x - sigma)**2 + (y - eta)**2)
        fres = U_initial * np.exp(1j * k * r_01) / (r_01**2)
        fres = np.sum(fres, axis=0)*ds
        fres = np.sum(fres, axis=0)*ds
        U_propagated = ( z_ / (1j * wavelength) ) * fres

        ax.cla()
        ax.set_title("U_propagated: z:{}".format(z_))
        ax.pcolormesh(np.squeeze(x), np.squeeze(y), np.squeeze(np.abs(U_propagated)**2))
        plt.pause(0.01)

    plt.show()




def create_pattern(Nx, Ny):
    """
    create a phase amplitude pattern
    """



def fraunhoffer_diffraction(object, z):

    # wavelength = 500*1e-9
    k = 2*np.pi / object.wavelength
    # z = 100 # meter

    U_prop = object.U_initial*np.exp(-1j * (2*np.pi / (object.wavelength*z)) * (object.x * object.sigma + object.y * object.eta))
    # integrate over surface
    U_prop = np.sum(U_prop, axis=0)*object.ds
    U_prop = np.sum(U_prop, axis=0)*object.ds

    plt.figure()
    plt.pcolormesh(np.squeeze(np.abs(object.U_initial)))
    plt.figure()
    plt.pcolormesh(np.squeeze(np.abs(U_prop)))
    plt.show()




if __name__ == "__main__":

    # fresnel_diffraction()
    object = DiffractionObject(ds=0.4e-3, y_max=1e-2, x_max=1e-2, wavelength=500e-9)
    # define initial field
    object.U_initial = np.zeros_like(object.sigma * object.eta, dtype=np.complex128)
    object.U_initial[:,18:20] = 1
    object.U_initial[:,30:32] = 1

    fraunhoffer_diffraction(object, z=100)


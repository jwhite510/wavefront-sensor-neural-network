import numpy as np
import matplotlib.pyplot as plt


class DiffractionObject():
    def __init__(self, N, x_max, wavelength):

        # object = DiffractionObject(ds=0.4e-3, d_max=1e-2, wavelength=500e-9)

        # measured diffraction pattern plane
        self.x_max = x_max

        # surface coordinates before propagation
        self.N = N

        self.dx = (2 * self.x_max) / self.N


        # define position axis
        self.x = np.arange(-self.N/2, self.N/2, 1)
        print("self.x =>", self.x)
        exit()

        self.wavelength = wavelength
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


def fraunhoffer_diffraction_fft(object, z):

    # fft object
    import ipdb; ipdb.set_trace() # BREAKPOINT
    print("BREAKPOINT")







if __name__ == "__main__":

    # grid space of the diffraction pattern
    N = 20 # measurement points
    diffraction_plane_x_max = 1 # meters
    diffraction_plane_z = 10 # meters
    wavelength = 400e-9

    # measured diffraction plane
    diffraction_plane_dx = 2*diffraction_plane_x_max/N
    diffraction_plane_x = diffraction_plane_dx * np.arange(-N/2, N/2, 1)

    # convert distance to frequency domain
    diffraction_plane_fx = diffraction_plane_x / (wavelength * diffraction_plane_z)
    diffraction_plane_d_fx = diffraction_plane_dx / (wavelength * diffraction_plane_z)

    # x coordinates at object plane
    object_plane_dx = 1 / ( diffraction_plane_d_fx * N)
    object_plane_x = object_plane_dx * np.arange(-N/2, N/2, 1)

    # # convert to image domain
    # print("d_f_x =>", d_f_x)
    # print("f_x[1] - f_x[0] =>", f_x[1] - f_x[0])






import numpy as np
import matplotlib.pyplot as plt


def make_object(N):
    obj = np.zeros((N,N), dtype=np.complex128)

    # generate random indexes
    indexes_n = np.random.randint(4,8)
    # for each index generate an x and y point
    x = []
    y = []
    for i in range(indexes_n):
        x.append(int(np.random.rand(1)*N))
        y.append(int(np.random.rand(1)*N))
        print("i =>", i)
        print("y =>", y)
        print("x =>", x)

    x.append(x[0])
    y.append(y[0])

    plt.figure()
    print("x =>", x)
    print("y =>", y)
    plt.plot(x, y)
    plt.show()


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
    diffraction_plane_dfx = diffraction_plane_dx / (wavelength * diffraction_plane_z)

    # x coordinates at object plane
    object_plane_dx = 1 / ( diffraction_plane_dfx * N)
    object_plane_x = object_plane_dx * np.arange(-N/2, N/2, 1)


    # construct object in the object plane
    object = make_object(N)








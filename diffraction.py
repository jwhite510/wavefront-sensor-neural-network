import numpy as np
import matplotlib.pyplot as plt

def orientation(p, q, r):
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])

    if val == 0:
        return 0
    elif val > 0:
        return 1
    else:
        return 2


def check_intersect(line1, line2):
    plt.figure(88)
    plt.gca().cla()
    plt.plot(line1[0], line1[1])
    plt.plot(line2[0], line2[1])

    p1 = (line1[0][0], line1[1][0])
    q1 = (line1[0][1], line1[1][1])
    p2 = (line2[0][0], line2[1][0])
    q2 = (line2[0][1], line2[1][1])
    # .. = (x, y)

    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    if o1 != o2 and o3 != o4:
        # return True
        print("true")
    plt.ioff()
    plt.show()
    exit()
    return False


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
        # print("i =>", i)
        # print("y =>", y)
        # print("x =>", x)

    x.append(x[0])
    y.append(y[0])

    # plt.figure()
    # print("x =>", x)
    # print("y =>", y)
    # plt.plot(x, y)

    # define test line
    test_linex = [3, 20]
    test_liney = [10, 10]
    plt.plot(test_linex, test_liney)
    # check if lines intersect
    # plt.ion()
    for i in range(0,len(x)-1):
        # plt.plot([x[i], x[i+1]], [y[i], y[i+1]])
        # plt.pause(1.1)
        check_intersect( [test_linex, test_liney], [[x[i], x[i+1]], [y[i], y[i+1]]])
        # x[i]
        # y[i]
        # x[i+1]
        # y[i+1]
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








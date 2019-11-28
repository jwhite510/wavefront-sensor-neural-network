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
        return 1.0
    return 0.0


def make_object(N):
    obj = np.zeros((N,N), dtype=np.complex128)

    # generate random indexes
    np.random.seed(3356)
    indexes_n = np.random.randint(4,8)
    # for each index generate an x and y point
    x = []
    y = []
    for i in range(indexes_n):
        x.append(int(np.random.rand(1)*N))
        y.append(int(np.random.rand(1)*N))

    x.append(x[0])
    y.append(y[0])

    plt.figure(1)
    plt.plot(x, y)
    plt.xlim(0,N)
    plt.ylim(0,N)
    plt.ion()

    # for each point on the grid
    for x_i in range(0, N):
        for y_i in range(0, N):
            # define test line
            test_linex = [x_i, N-1]
            test_liney = [y_i, y_i]

            # for each line segment
            intersections = 0.0
            intersected_vertices = []
            for i in range(0,len(x)-1):
                # count the number of intersections
                edge_y = [y[i], y[i+1]]
                edge_x = [x[i], x[i+1]]

                intersections += check_intersect( [test_linex, test_liney], [edge_x, edge_y])
                # dont count vertices twice

                # check if on a vertice
                if test_liney[0] in edge_y:

                    # get the point
                    # index_pt = edge_y.index(test_liney[0])
                    point_vertex = [edge_x, edge_y]
                    # append this point
                    intersected_vertices.append(point_vertex)

            if len(intersected_vertices) > 0:
                # check for intersections that are vertically allgined
                print("intersected_vertices =>", intersected_vertices)
                for vertices in intersected_vertices:
                    print(vertices)
                # import ipdb; ipdb.set_trace() # BREAKPOINT
                # print("BREAKPOINT")
                pass
                # import ipdb; ipdb.set_trace() # BREAKPOINT
                # print("BREAKPOINT")

            plt.figure(2)
            plt.gca().cla()
            plt.pcolormesh(np.abs(obj))
            plt.figure(1)
            plt.gca().cla()
            plt.plot(x, y)
            plt.plot(test_linex, test_liney)
            plt.xlim(0,N)
            plt.ylim(0,N)

            plt.pause(0.10)
            # even number of intersections:
            # intersections -= (len(intersected_vertices) / 2)
            print("intersections =>", intersections)
            if intersections % 2 == 0:
                obj[y_i, x_i] = 1

    plt.figure(2)
    plt.pcolormesh(np.abs(obj))
    plt.show()






    plt.show()



if __name__ == "__main__":

    # grid space of the diffraction pattern
    N = 40 # measurement points
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








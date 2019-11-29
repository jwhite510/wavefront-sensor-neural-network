import numpy as np
import math
from PIL import Image, ImageDraw
from PIL import ImagePath
import matplotlib.pyplot as plt


def check_if_vertical(line1, line2):
    # find common point between two lines
    common_point = [line1[0][0], line1[1][0]]
    line1_point = [line1[0][1], line1[1][1]]
    if line2[0][0] != common_point[0] or line2[1][0] != common_point[1]:
        if line2[0][1] != common_point[0] or line2[1][1] != common_point[1]:
            common_point = [line1[0][1], line1[1][1]]
            line1_point = [line1[0][0], line1[1][0]]
    line2_point = [line2[0][1], line2[1][1]]
    if common_point == line2_point:
        line2_point = [line2[0][0], line2[1][0]]

    line2_point[1] -= common_point[1]
    line1_point[1] -= common_point[1]

    if line2_point[1] > 0 and line1_point[1] > 0:
        return 0.0
    elif line2_point[1] < 0 and line1_point[1] < 0:
        return 0.0
    else:
        # print("is vertical, subtracting")
        return 1.0




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

    seedval = np.random.randint(1,999)
    print("seedval =>", seedval)
    seedval = 454
    np.random.seed(seedval)

    obj = np.zeros((N,N), dtype=np.complex128)

    # generate random indexes
    # np.random.seed(3357)
    indexes_n = np.random.randint(4,8)
    # for each index generate an x and y point
    x = []
    y = []
    for i in range(indexes_n):
        x.append(int(np.random.rand(1)*N))
        y.append(int(np.random.rand(1)*N))

    x.append(x[0])
    y.append(y[0])

    # import ipdb; ipdb.set_trace() # BREAKPOINT
    # print("BREAKPOINT")
    scalef = 9
    xy = [(scalef*x_, scalef*y_) for x_, y_ in zip(x,y)]

    image = ImagePath.Path(xy).getbbox()
    size = list(map(int, map(math.ceil, image[2:])))
    # img = Image.new("RGB", size, "#f9f9f9")
    img = Image.new("RGB", size, "#ffffff")
    img1 = ImageDraw.Draw(img)
    img1.polygon(xy, fill ="#000000")
    # img1.polygon(xy, fill ="#eeeeff", outline ="blue")
    img.show()
    # exit()

    plt.figure(1)
    plt.plot(x, y)
    plt.xlim(0,N)
    plt.ylim(0,N)
    # for each point on the grid
    plt.ion()
    for x_i in range(0, N):
        for y_i in range(0, N):
            # define test line
            test_linex = [x_i, N-1]
            test_liney = [y_i, y_i]

            # for each line segment
            intersections = 0.0
            intersected_vertices = []
            roll = False
            for i in range(0,len(x)-1):
                # count the number of intersections
                edge_y = [y[i], y[i+1]]
                edge_x = [x[i], x[i+1]]

                intersections += check_intersect( [test_linex, test_liney], [edge_x, edge_y])
                # dont count vertices twice

                # check if on a vertice
                if test_liney[0] in edge_y:
                    # if the line
                    index_ = edge_y.index(test_liney[0])
                    if test_linex[0] < edge_x[index_]:
                    # if True:
                        # get the point
                        # index_pt = edge_y.index(test_liney[0])
                        point_vertex = [edge_x, edge_y]
                        # append this point
                        intersected_vertices.append(point_vertex)
                        if i == 0:
                            roll = True

            if roll:
                intersected_vertices.insert(0,intersected_vertices.pop())

            if len(intersected_vertices) > 0:
                # check that vertices are in order -- edge case
                # check for intersections that are vertically allgined
                print("intersected_vertices =>", intersected_vertices)
                for line1_, line2_ in zip(intersected_vertices[0::2], intersected_vertices[1::2]):
                    intersections -= check_if_vertical(line1_, line2_)
                    print(check_if_vertical(line1_, line2_))

            if intersections % 2 == 0:
                obj[y_i, x_i] = 1

            # if x_i > 3 and y_i>12:
                # plt.figure(1)
                # plt.gca().cla()
                # plt.plot(x,y)
                # plt.xlim(0,40)
                # plt.ylim(0,40)
                # plt.plot(test_linex, test_liney)
                # plt.figure(2)
                # plt.pcolormesh(np.abs(obj))
                # plt.pause(0.1)


    plt.ioff()
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








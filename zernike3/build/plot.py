import numpy as np
import matplotlib.pyplot as plt

def readarray(filename):

    shape = None
    datatype = None
    total_elems = 1

    # determine size and shape
    with open(filename, "r") as file:
        for linenum, line in enumerate(file.readlines()):
            # determine data type
            if linenum == 1:
                if line.strip() == "float":
                    datatype = np.float
                if line.strip() == "complex_float":
                    datatype = np.complex64

            # determine shape
            elif linenum == 3:
                shape = line.split()
                shape = [int(s) for s in shape]
                for e in shape:
                    total_elems *= e
                shape = tuple(shape)
                break
    arr = np.zeros((int(total_elems)), dtype=datatype)
    index = 0
    with open(filename, "r") as file:
        for linenum, line in enumerate(file.readlines()):

            if linenum > 4:

                if datatype == np.float:
                    arr[index] = float(line)

                elif datatype == np.complex64:
                    arr[index] = complex(line)

                index += 1
    arr = arr.reshape(shape)

    return arr


if __name__ == "__main__":

    arr = readarray("zernike_polynom_com.txt")

    # plt.figure()
    # plt.pcolormesh(arr)
    print(arr)
    # plt.colorbar()
    # plt.show()

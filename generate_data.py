import numpy as np
import matplotlib.pyplot as plt
import tables
import diffraction_functions


def make_dataset(filename, N, samples):
    # create the tables file

    with tables.open_file(filename, "w") as hdf5file:

        # create array for the object
        hdf5file.create_earray(hdf5file.root, "object_amplitude", tables.Float64Atom(), shape=(0,N*N))

        # create array for the object phase
        hdf5file.create_earray(hdf5file.root, "object_phase", tables.Float64Atom(), shape=(0,N*N))

        # create array for the image
        hdf5file.create_earray(hdf5file.root, "diffraction", tables.Float64Atom(), shape=(0,N*N))

        hdf5file.create_earray(hdf5file.root, "N", tables.Int32Atom(), shape=(0,1))

        hdf5file.close()


    with tables.open_file(filename, mode='a') as hd5file:

        # save the dimmensions of the data
        hd5file.root.N.append(np.array([[N]]))
        for i in range(samples):

            if i % 100 == 0:
                print("Generating sample %i of %i" % (i, samples))


            # generate a sample
            object, object_phase = diffraction_functions.make_object(N, min_indexes=4, max_indexes=8)
            object_with_phase = diffraction_functions.make_object_phase(object, object_phase)
            # remove ambiguity

            object_with_phase = diffraction_functions.remove_ambiguitues(object_with_phase)

            object_amplitude = np.abs(object_with_phase)

            object_phase = np.angle(object_with_phase)
            object_phase[object_amplitude<0.2] = 0

            diffraction_pattern = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(object_with_phase)))

            # absolute value
            diffraction_pattern = np.abs(diffraction_pattern)

            # normalize the diffraction pattern
            diffraction_pattern = diffraction_pattern / np.max(diffraction_pattern)

            hd5file.root.object_amplitude.append(object_amplitude.reshape(1,-1))
            hd5file.root.object_phase.append(object_phase.reshape(1,-1))
            hd5file.root.diffraction.append(diffraction_pattern.reshape(1,-1))


if __name__ == "__main__":
    # generate a data set
    N = 40

    make_dataset("train_data.hdf5", N=N, samples=40000)

    make_dataset("test_data.hdf5", N=N, samples=200)

    # test open the data set
    index = 4
    with tables.open_file("train_data.hdf5", mode="r") as hdf5file:

        # print("hdf5file.root.N =>", hdf5file.root.N[0,0])
        N = hdf5file.root.N[0,0]

        object = hdf5file.root.object_amplitude[index,:].reshape(N,N)
        object_phase = hdf5file.root.object_phase[index,:].reshape(N,N)
        diffraction = hdf5file.root.diffraction[index,:].reshape(N,N)

        plt.figure(1)
        plt.pcolormesh(object)
        plt.figure(2)
        plt.pcolormesh(object_phase)
        plt.figure(3)
        plt.pcolormesh(diffraction)
        plt.show()



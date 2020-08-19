import numpy as np
import tables
import matplotlib.pyplot as plt


if __name__ == "__main__":

    FILENAME1="test_round.hdf5"
    FILENAME2="test_square.hdf5"
    file1 = tables.open_file(FILENAME1, mode="r")
    file2 = tables.open_file(FILENAME2, mode="r")

    N = file1.root.N[0,0]
    index=1


    for index in range(20,50):
        object_real_samples1 = file1.root.object_real[index, :].reshape(N,N)
        object_imag_samples1 = file1.root.object_imag[index, :].reshape(N,N)
        diffraction_samples1 = file1.root.diffraction[index, :].reshape(N,N)

        object_real_samples2 = file2.root.object_real[index, :].reshape(N,N)
        object_imag_samples2 = file2.root.object_imag[index, :].reshape(N,N)
        diffraction_samples2 = file2.root.diffraction[index, :].reshape(N,N)

        fig,ax=plt.subplots(1,3,figsize=(10,5))
        fig.suptitle("ROUND")
        ax[0].imshow(object_real_samples1)
        ax[1].imshow(object_imag_samples1)
        ax[2].imshow(diffraction_samples1)

        fig,ax=plt.subplots(1,3,figsize=(10,5))
        fig.suptitle("SQUARE")
        ax[0].imshow(object_real_samples2)
        ax[1].imshow(object_imag_samples2)
        ax[2].imshow(diffraction_samples2)

        plt.show()

    file1.close()
    file2.close()


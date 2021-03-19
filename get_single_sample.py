import numpy as np
import tables
import pickle


if __name__ == "__main__":
    hdf5_file_validation = tables.open_file("zernike3/build/test.hdf5", mode="r")
    sample = hdf5_file_validation.root.diffraction[0, :].reshape(256,256)
    with open("sample.p", "wb") as file:
        pickle.dump(sample,file)

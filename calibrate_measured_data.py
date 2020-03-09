import numpy as np
from  astropy.io import fits
import tables
import matplotlib.pyplot as plt


if __name__ == "__main__":

    fits_file_name = "/home/jonathon/Documents/test/windowshare/1.fits"
    thing = fits.open(fits_file_name)
    print(thing.info())
    # print("thing[0].data =>", thing[0].data)
    plt.figure()

    print("np.shape(thing[0].data) =>", np.shape(thing[0].data))
    plt.imshow(thing[0].data[0,:,:])
    plt.show()






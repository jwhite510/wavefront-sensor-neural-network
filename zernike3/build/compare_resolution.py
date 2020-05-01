import numpy as np
import time
import pickle
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import PIL.ImageOps
import diffraction_functions
import tables


if __name__ == "__main__":
    with open("1024pickle.p", "rb") as file:
        arr1024 = pickle.load(file)

    with open("128pickle.p", "rb") as file:
        arr128 = pickle.load(file)

    print("np.shape(arr128) => ",np.shape(arr128))
    print("np.shape(arr1024) => ",np.shape(arr1024))

    arr = arr1024
    fig = plt.figure(figsize=(10,5))
    gs = fig.add_gridspec(1,2)

    ax = fig.add_subplot(gs[0,0])
    ax.set_title("object (1024)")
    ax.imshow(np.abs(arr))

    # fourier transform
    diffraction = np.abs(np.fft.fftshift(np.fft.fft2(np.fft.fftshift(arr))))**2

    ax = fig.add_subplot(gs[0,1])
    ax.set_title("diffraction pattern (1024)")
    ax.imshow(diffraction)


    plt.show()




import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    c_r=np.loadtxt("complex_object_real.dat")
    c_i=np.loadtxt("complex_object_imag.dat")
    c=c_r + 1j * c_i
    plt.figure()
    plt.imshow(np.abs(c))
    plt.figure()
    plt.imshow(np.angle(c))
    plt.show()

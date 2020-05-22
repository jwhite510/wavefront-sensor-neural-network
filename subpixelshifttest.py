import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":

    plt.ion()
    fig,ax=plt.subplots(2,2, figsize=(10,10))
    figt=fig.text(0.5, 0.95, "title", ha="center", size=30)

    for shift in range(0,256):
        # fig.clf()
        a_x = np.zeros((128,128))
        # a_x[500:524,500:524]=1
        # a_x[60+shift:68+shift,60:68]=1
        # a_x[60:68,60+shift:68+shift]=1
        a_x[60:68,60:68]=1

        figt.set_text(str(shift))

        a_f=np.fft.fftshift(np.fft.fft2(np.fft.fftshift(a_x)))
        a_f_pad=np.pad(a_f,shift,mode="constant")
        a_x_pad=np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(a_f_pad)))

        ax[0,0].cla()
        ax[0,0].pcolormesh(np.real(a_f))
        ax[0,0].set_title("np.real(a_f)")

        ax[0,1].cla()
        ax[0,1].pcolormesh(np.real(a_x))
        ax[0,1].set_title("np.real(a_x)")

        ax[1,0].cla()
        ax[1,0].pcolormesh(np.real(a_f_pad))
        ax[1,0].set_title("np.real(a_f_pad)")

        ax[1,1].cla()
        ax[1,1].pcolormesh(np.real(a_x_pad))
        ax[1,1].set_title("np.real(a_x_pad)")

        plt.pause(0.1)
        # plt.show()

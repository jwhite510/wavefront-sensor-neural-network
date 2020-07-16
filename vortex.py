import numpy as np
import matplotlib.pyplot as plt

def random_beam(x,y,N):
    E = np.zeros((x.size,y.size))
    p1 = -1
    p2 = 1
    width_min = 0.1
    width_max = 3.0
    for _ in range(N):

        a=np.random.rand()
        _x=p1+np.random.rand()*(p2-p1)
        _y=p1+np.random.rand()*(p2-p1)
        _wx = width_min+np.random.rand()*(width_max-width_min)
        _wy = width_min+np.random.rand()*(width_max-width_min)
        # E +=a*np.exp((-(x-_x)/_wx)**2)*np.exp((-(y-_y)/_wy)**2)
        # E +=a*np.exp(-(x-_x)**2)*np.exp(-(y-_y)**2)
        E +=a*np.exp(-(((x-_x)/_wx)**2))*np.exp(-(((y-_y)/_wy)**2))

    return E

# def random_beam(x,y,N):
    # E = np.zeros((x.size,y.size))
    # p1 = -5
    # p2 = 5
    # width_min = 0.0
    # width_max = 3.0
    # for _ in range(N):

        # a=np.random.rand()
        # _x=p1+np.random.rand()*(p2-p1)
        # _y=p1+np.random.rand()*(p2-p1)
        # E +=a*np.exp(-(x-_x)**2)*np.exp(-(y-_y)**2)

    # return E

if __name__ == "__main__":

    x = np.linspace(-10,10,300).reshape(1,-1)
    y = np.linspace(-10,10,300).reshape(-1,1)
    r = np.sqrt(x**2 + y**2)
    # angle
    # phi = 

    # construct angle

    vortex_offset = (0.0,0.0)

    phi = np.arctan2(y+vortex_offset[1],x+vortex_offset[0])
    m = 3

    E = np.exp(-x**2)*np.exp(-y**2)
    np.random.seed(10)
    E = random_beam(x,y,7)

    E = np.array(E,dtype=np.complex128)
    E*= np.exp(1j * m * phi)


    fig,ax=plt.subplots(2,2,figsize=(10,10))
    fig.suptitle("Vortex Beam")
    ax[0][0].imshow(np.abs(E)**2)
    ax[0][0].text(0.1, 0.9,"Initial Amplitude", fontsize=10, ha='center', transform=ax[0][0].transAxes, backgroundcolor="yellow")

    ax[0][1].imshow(np.angle(E))
    ax[0][1].text(0.1, 0.9,"Initial Phase", fontsize=10, ha='center', transform=ax[0][1].transAxes, backgroundcolor="yellow")
    ax[0][1].text(0.1, 0.1,
            r"$exp(i m \phi)$"+"\n"+
            "phase offset : ("+str(vortex_offset[0])+","+str(vortex_offset[1])+")\n"
            "m="+str(m)
            , fontsize=10, ha='center', transform=ax[0][1].transAxes, backgroundcolor="yellow")

    ax[1][0].imshow(np.abs(np.fft.fftshift(np.fft.fft2(np.fft.fftshift(E)))))
    ax[1][0].text(0.1, 0.9,"Propagated (fft) Amplitude", fontsize=10, ha='center', transform=ax[1][0].transAxes, backgroundcolor="yellow")

    ax[1][1].imshow(np.angle(np.fft.fftshift(np.fft.fft2(np.fft.fftshift(E)))))
    ax[1][1].text(0.1, 0.9,"Propagated (fft) Phase", fontsize=10, ha='center', transform=ax[1][1].transAxes, backgroundcolor="yellow")
    for i in range(0,2):
        for j in range(0,2):
            ax[i][j].set_xlim(100,200)
            ax[i][j].set_ylim(100,200)


    # plt.figure()
    # plt.imshow(np.abs(E)**2)

    # plt.figure()
    # plt.imshow(np.angle(E))

    # plt.figure()
    # plt.imshow(np.abs(np.fft.fftshift(np.fft.fft2(np.fft.fftshift(E)))))
    # plt.figure()
    # plt.imshow(np.angle(np.fft.fftshift(np.fft.fft2(np.fft.fftshift(E)))))

    plt.show()



import numpy as np
import matplotlib.pyplot as plt


class Params():
    def __init__(self):
        self.beta_Ta=None
        self.delta_Ta=None
        self.dz=None
        self.lam=None
        self.k=None

def loadparams(filename,obj):
    tmp_params={}
    with open(filename) as file:
        for line in file.readlines():
            linesplit=line.replace(":","").split()
            tmp_params[linesplit[0]]=float(linesplit[1])

    obj.beta_Ta=tmp_params["beta_Ta"]
    obj.delta_Ta=tmp_params["delta_Ta"]
    obj.dz=tmp_params["dz"]
    obj.lam=tmp_params["lam"]
    obj.k=tmp_params["k"]

class Propagate():
    def __init__(self):
        self.steps_Si=None
        with open("steps_Si.dat","r") as file:
            self.steps_Si=int(file.readlines()[0])
        self.steps_cu=None
        with open("steps_cu.dat","r") as file:
            self.steps_cu=int(file.readlines()[0])

        self.params_Si=Params()
        loadparams("params_Si.dat",self.params_Si)
        self.params_cu=Params()
        loadparams("params_cu.dat",self.params_cu)

        self.wf_f=None
        with open("wfs_f.dat","r") as file:
            self.wf_f=np.array(file.readlines(),dtype=np.double)

def plot_diffraction(arr):
    fig, ax = plt.subplots(1,2, figsize=(15,5))
    fig.suptitle(arr)
    arr=np.loadtxt(arr)
    ax[0].pcolormesh(np.abs(arr))
    ax[1].pcolormesh(np.abs(np.fft.fftshift(np.fft.fft2(np.fft.fftshift(arr))))**2)

if __name__ == "__main__":
    propagate=Propagate()
    # plt.show()





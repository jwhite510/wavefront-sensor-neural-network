import numpy as np
import diffraction_functions
import matplotlib.pyplot as plt
import tensorflow as tf


def read_complex_array(filenameprefix):
    a=np.loadtxt(filenameprefix+"_real.dat")
    b=np.loadtxt(filenameprefix+"_imag.dat")
    z=a+1j*b
    return z



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

        self.slice_Si=read_complex_array("slice_Si")
        self.slice_cu=read_complex_array("slice_cu")
        self.wavefront=tf.placeholder(tf.complex64,shape=[None,128,128,1])
        self.through_wf=self.setup_graph_through_wfs()

    def setup_graph_through_wfs(self):

        wavefront_ref=self.wavefront
        for _ in range(self.steps_Si):
            wavefront_ref=forward_propagate(wavefront_ref,self.slice_Si,self.wf_f,self.params_Si)
        for _ in range(self.steps_cu):
            wavefront_ref=forward_propagate(wavefront_ref,self.slice_cu,self.wf_f,self.params_cu)
        return wavefront_ref

def forward_propagate(E,slice,f,p):
    slice=np.expand_dims(slice,0)
    slice=np.expand_dims(slice,3)

    E*=slice
    E=diffraction_functions.tf_fft2(E,dimmensions=[1,2])
    gamma1=tf.constant(
                1-
                (p.lam*f.reshape(-1,1))**2-
                (p.lam*f.reshape(1,-1))**2,
                dtype=tf.complex64
            )
    gamma=tf.sqrt(
            gamma1
            )
    k_sq = 2 * np.pi * p.dz / p.lam
    H = tf.exp(
            tf.complex(
                real=tf.zeros_like(tf.real(gamma)*k_sq),
                imag=tf.real(gamma)*k_sq
                )
            )*tf.exp(
            tf.complex(
                real=-1*tf.imag(gamma)*k_sq,
                imag=tf.zeros_like(tf.imag(gamma)*k_sq)
                )
            )
    H = tf.expand_dims(H,0)
    H = tf.expand_dims(H,-1)
    E*=H
    E=diffraction_functions.tf_ifft2(E,dimmensions=[1,2])
    return E


def plot_diffraction(arr,name):
    fig, ax = plt.subplots(1,2, figsize=(15,5))
    fig.suptitle(name)
    ax[0].pcolormesh(np.abs(arr))
    ax[1].pcolormesh(np.abs(np.fft.fftshift(np.fft.fft2(np.fft.fftshift(arr))))**2)

if __name__ == "__main__":
    propagate=Propagate()
    wavefront_before=read_complex_array("interped_arr_before")
    wavefront_before=np.expand_dims(wavefront_before,0)
    wavefront_before=np.expand_dims(wavefront_before,-1)
    with tf.Session() as sess:
        out=sess.run(propagate.through_wf,feed_dict={propagate.wavefront:wavefront_before})
        plt.figure()
        plt.imshow(np.squeeze(np.abs(out)))

    # propagate.through_wfs(wavefront_before)
    interped_arr_before=read_complex_array("interped_arr_before")
    plot_diffraction(interped_arr_before,"interped_arr_before")

    interped_arr_after=read_complex_array("interped_arr_after")
    plot_diffraction(interped_arr_after,"interped_arr_after")

    plot_diffraction(np.squeeze(out),"out")

    plt.show()





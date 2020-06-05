import numpy as np
import os
import matplotlib.pyplot as plt
from zernike3.build import PropagateTF
import diffraction_functions
import tensorflow as tf

def create_aperture_material(N,x,p):

    aperture_radius=2.7e-6
    material=np.zeros((N,N),dtype=np.complex64)
    x=x.reshape(1,-1)
    y=x.reshape(-1,1)
    r = np.sqrt(x**2 + y**2)

    material[r<aperture_radius]=1.0+0j
    material[r>=aperture_radius]=np.exp(-p.k*p.beta_Ta*p.dz)*np.exp(-1j*p.k*p.delta_Ta*p.dz)
    return material


if __name__ == "__main__":

    N=128
    wavefront = tf.placeholder(tf.complex64, shape=[1,N,N,1])

    params_Si=PropagateTF.Params()
    PropagateTF.loadparams("zernike3/build/params_Si.dat",params_Si)

    params_cu=PropagateTF.Params()
    PropagateTF.loadparams("zernike3/build/params_cu.dat",params_cu)

    # get position axis of wavefront sensor
    measured_axes, amplitude_mask=diffraction_functions.get_amplitude_mask_and_imagesize(N,int(N/2))
    x=measured_axes["object"]["x"]

    # this is the same data contained in measured_axes, but im loading it from the dat file anyway
    wf_f=None
    with open(os.path.join("zernike3/build/wfs_f.dat"),"r") as file:
        wf_f=np.array(file.readlines(),dtype=np.double)

    # distance of spherical aperture
    cu_distance=300e-9
    si_distance=50e-9
    steps_cu=int(round(cu_distance/params_cu.dz))
    steps_Si=int(round(si_distance/params_Si.dz))

    # steps_Si=None
    # with open(os.path.join("zernike3/build/steps_Si.dat"),"r") as file:
        # steps_Si=int(file.readlines()[0])
    # steps_cu=None
    # with open(os.path.join("zernike3/build/steps_cu.dat"),"r") as file:
        # steps_cu=int(file.readlines()[0])

    # create the material
    slice_Si=create_aperture_material(N,x,params_Si)
    slice_cu=create_aperture_material(N,x,params_cu)

    # test the wavefront sensor
    # slice_Si=PropagateTF.read_complex_array("zernike3/build/slice_Si")
    # slice_cu=PropagateTF.read_complex_array("zernike3/build/slice_cu")


    # increasing the wavelength will show the effect of propagation
    # params_Si.lam*=20
    # params_cu.lam*=20

    propagated=wavefront
    for _ in range(steps_Si):
        propagated=PropagateTF.forward_propagate(propagated,slice_Si,wf_f,params_Si)
    for _ in range(steps_cu):
        propagated=PropagateTF.forward_propagate(propagated,slice_cu,wf_f,params_cu)


    with tf.Session() as sess:
        out=sess.run(propagated,feed_dict={wavefront:np.ones((N,N)).reshape(1,N,N,1)})

    # plt.figure()
    # plt.imshow(np.squeeze(np.real(out)))
    plt.figure()
    plt.imshow(np.squeeze(np.abs(out)))
    plt.show()

    print("ran!")
    print(out)



    # 300 nm Cu

    # 50 nm Si




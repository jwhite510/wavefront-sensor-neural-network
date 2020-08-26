import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
import diffraction_functions



def makezernike(m: int,n: int, N_computational: int)->np.array:
    x = np.linspace(-7,7,N_computational).reshape(-1,1)
    y = np.linspace(-7,7,N_computational).reshape(1,-1)
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(x,y)

    positive_m = False
    if m >= 0:
        positive_m=True
    m = abs(m)
    zernike_polynom = np.zeros((N_computational,N_computational))
    k=0
    while k <= (n-m)/2:

        numerator = -1 ** k
        numerator *= math.factorial(n-k)

        denominator = math.factorial(k)
        denominator *= math.factorial(((n+m)/2)-k)
        denominator *= math.factorial(((n-m)/2)-k)

        scalar = numerator/denominator

        zernike_polynom+=scalar*rho**(n-2*k)
        k+=1

    if positive_m:
        zernike_polynom *= np.cos(m * phi)
    else:
        zernike_polynom *= np.sin(m * phi)

    # set values outside unit circle to 0
    zernike_polynom[rho>1]=0
    return zernike_polynom

class Zernike_C():
    def __init__(self,m:int,n:int):
        self.m=m
        self.n=n

class Params():
  beta_Ta=None
  delta_Ta=None
  dz = 10e-9
  lam = 633e-9
  k = 2 * np.pi / lam

def create_slice(p: Params, N_interp: int)->np.array:
    _, wfs = diffraction_functions.get_amplitude_mask_and_imagesize(N_interp, N_interp//3)
    slice = np.zeros((N_interp,N_interp),dtype=np.complex128)
    slice[wfs<0.5]=np.exp(-1*p.k * p.beta_Ta * p.dz)*\
                    np.exp(-1j*p.k*p.delta_Ta*p.dz)
    slice[wfs>=0.5]=1.0
    return slice

class DataGenerator():
    def __init__(self,N_computational:int,N_interp:int,crop_size:int):
        self.N_interp=N_interp
        self.crop_size=crop_size
        self.N_computational=N_computational
        # generate zernike coefficients
        self.batch_size=1
        start_n=2
        max_n=4
        self.zernike_cvector = []

        for n in range(start_n,max_n+1):
            for m in range(n,-n-2,-2):
                self.zernike_cvector.append(Zernike_C(m,n))

        self.mn_polynomials=np.zeros((len(self.zernike_cvector),N_computational,N_computational))
        self.mn_polynomials_index=0
        for _z in self.zernike_cvector:
            z_arr = makezernike(_z.m,_z.n,N_computational)
            self.mn_polynomials[self.mn_polynomials_index,:,:]=z_arr
            self.mn_polynomials_index+=1

        # define materials
        # https://refractiveindex.info/?shelf=main&book=Cu&page=Johnson
        # https://refractiveindex.info/?shelf=main&book=Si3N4&page=Luke
        params_cu = Params()
        params_cu.delta_Ta = 0.26965-1 # double check this
        params_cu.beta_Ta = 3.4106
        params_Si = Params()
        params_Si.delta_Ta = 2.0394-1
        params_Si.beta_Ta = 0.0
        slice_cu = create_slice(params_cu,N_interp)
        slice_Si = create_slice(params_Si,N_interp)

        Si_distance = 50e-9;
        cu_distance = 150e-9;
        steps_Si = round(Si_distance / params_Si.dz);
        steps_cu = round(cu_distance / params_cu.dz);

        # propagator through wavefront sensor
        propagate_tf=PropagateTF(N_interp,steps_Si,params_Si,slice_Si,steps_cu,params_cu,slice_cu)
        self.x = tf.placeholder(tf.complex64, shape=[self.batch_size, 128 , 128, 1])
        self.prop=propagate_tf.setup_graph_through_wfs(self.x)
        self.buildgraph()

    def buildgraph(self):

        # scalars for zernike coefficients
        self.x = tf.placeholder(tf.float32, shape=[self.batch_size, len(self.zernike_cvector)])

        self.tf_mn_polynomials = tf.constant(self.mn_polynomials,dtype=tf.float32)

        self.zernike_polynom = tf.zeros([self.batch_size, self.N_computational,self.N_computational],dtype=tf.float32)

        for i in range(int(self.tf_mn_polynomials.shape[0])):
            self.zernike_polynom+=self.x[:,i]*self.tf_mn_polynomials[i,:,:]

        with tf.Session() as sess:
            out = sess.run(self.zernike_polynom,feed_dict={self.x:np.array([[1,0,0,0,0,0,0,0,0,0,0,0]])})
            plt.figure()
            plt.imshow(np.squeeze(out))

            out = sess.run(self.zernike_polynom,feed_dict={self.x:np.array([[1,1,0,0,0,0,0,0,0,0,0,0]])})
            plt.figure()
            plt.imshow(np.squeeze(out))

            out = sess.run(self.zernike_polynom,feed_dict={self.x:np.array([[1,1,1,1,1,1,0,0,0,0,0,0]])})
            plt.figure()
            plt.imshow(np.squeeze(out))

            plt.show()



    # def makesample()->np.array:



class PropagateTF():
    def __init__(self, N_interp:int, steps_Si:int, params_Si:Params, slice_Si:np.array, steps_cu:int, params_cu:Params, slice_cu:np.array):

        self.steps_Si=steps_Si
        self.steps_cu=steps_cu
        self.params_Si=params_Si
        self.params_cu=params_cu
        self.slice_Si=slice_Si
        self.slice_cu=slice_cu

        measured_axes, _ = diffraction_functions.get_amplitude_mask_and_imagesize(N_interp, N_interp//3)
        self.wf_f = measured_axes["diffraction_plane"]["f"]

    def setup_graph_through_wfs(self, wavefront):

        wavefront_ref=wavefront
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
if __name__ == "__main__":
    datagenerator = DataGenerator(1024,128,200)
    pass

    # initialize tensorflow data generation


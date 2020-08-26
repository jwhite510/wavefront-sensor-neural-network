import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
import diffraction_functions


# def gaussian_propagate(zernike_polynom:tf.Tensor,scale:tf.Tensor)->tf.Tensor:


def tf_make_zernike(m:int, n:int, N_computational:int, scale:tf.Tensor, amp:tf.Tensor)->tf.Tensor:
    _scale = tf.expand_dims(scale,axis=-1)
    x = np.linspace(-7,7,N_computational).reshape(1,-1,1)
    y = np.linspace(-7,7,N_computational).reshape(1,1,-1)
    x = (1/_scale)*tf.constant(x,dtype=tf.float32)
    y = (1/_scale)*tf.constant(y,dtype=tf.float32)
    # x = scale * x
    # y = scale * y
    rho = tf.sqrt(x**2 + y**2)
    phi = tf.atan2(x,y)

    positive_m = False
    if m >= 0:
        positive_m=True
    m = abs(m)
    # summation over k
    rho = tf.expand_dims(rho,axis=1)
    # for each k value
    # k = np.linspace(0,(n-m)//2)
    k = np.arange(start=0,stop=1+((n-m)//2))
    numerator = (-1)**k
    for i in range(len(k)):
        numerator[i]*= math.factorial(n-k[i])

    denominator = np.ones_like(k)
    for i in range(len(k)):
        denominator[i]*= math.factorial(k[i])
        denominator[i]*= math.factorial(((n+m)/2)-k[i])
        denominator[i]*= math.factorial(((n-m)/2)-k[i])

    scalar = numerator/denominator
    zernike_polynom = scalar.reshape(1,-1,1,1) * rho**(n-2*k.reshape(1,-1,1,1))
    zernike_polynom = tf.reduce_sum(zernike_polynom,axis=1)

    if positive_m:
        zernike_polynom *= tf.cos(m * phi)
    else:
        zernike_polynom *= tf.sin(m * phi)

    # set values outside unit circle to 0
    zernike_polynom*=tf.cast(tf.less_equal(tf.squeeze(rho,axis=1),1),dtype=tf.float32)
    _amp = tf.expand_dims(amp,axis=-1)
    _amp = tf.expand_dims(_amp,axis=-1)
    zernike_polynom*=_amp
    return tf.expand_dims(zernike_polynom,axis=1)

def makezernike(m: int,n: int, N_computational: int)->np.array:
    x = np.linspace(-7/2,7/2,N_computational).reshape(-1,1)
    y = np.linspace(-7/2,7/2,N_computational).reshape(1,-1)
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
        self.batch_size=4
        start_n=1
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
        self.propagate_tf=PropagateTF(N_interp,steps_Si,params_Si,slice_Si,steps_cu,params_cu,slice_cu)
        self.buildgraph()


    def buildgraph(self):

        # scalars for zernike coefficients
        self.x = tf.placeholder(tf.float32, shape=[None, len(self.zernike_cvector)])
        # self._x = tf.expand_dims(self.x,axis=-1)
        # self._x = tf.expand_dims(self._x,axis=-1)


        self.scale = tf.placeholder(tf.float32, shape=[None,1])
        # generate polynomials
        zernikes=[]
        for i in range(len(self.zernike_cvector)):
            _z = self.zernike_cvector[i]
            zernikes.append(tf_make_zernike(_z.m,_z.n,self.N_computational,self.scale,self.x[:,i]))

        zernikes=tf.concat(zernikes,axis=1)
        self.zernike_polynom = tf.reduce_sum(zernikes,axis=1)

        # propagate through gaussian
        x = np.linspace(1,-1,self.N_computational).reshape(1,-1,1)
        y = np.linspace(1,-1,self.N_computational).reshape(1,1,-1)
        self._scale = tf.expand_dims(self.scale,axis=-1)
        x = (1/self._scale)*tf.constant(x,dtype=tf.float32)
        y = (1/self._scale)*tf.constant(y,dtype=tf.float32)
        width = 0.05
        self.gaussian_amp = tf.exp(-(x**2)/(width**2))*tf.exp(-(y**2)/(width**2))

        self.field = tf.complex(real=self.gaussian_amp,imag=tf.zeros_like(self.gaussian_amp)) * tf.exp(tf.complex(real=tf.zeros_like(self.zernike_polynom),imag=self.zernike_polynom))

        # fft
        self.field = tf.expand_dims(self.field,axis=-1)
        self.field_ft=diffraction_functions.tf_fft2(self.field,dimmensions=[1,2])

        # crop and interpolate
        self.field_cropped = self.field_ft[:,(self.N_computational//2)-(self.N_interp//2):(self.N_computational//2)+(self.N_interp//2),(self.N_computational//2)-(self.N_interp//2):(self.N_computational//2)+(self.N_interp//2),:]

        # TODO set phase to 0 at center
        z_center = self.field_cropped[:,self.N_interp//2,self.N_interp//2,0]
        z_center = tf.expand_dims(z_center,-1)
        z_center = tf.expand_dims(z_center,-1)
        z_center = tf.expand_dims(z_center,-1)
        self.field_cropped = self.field_cropped * tf.exp(tf.complex(real=0.0,imag=-1.0*tf.angle(z_center)))

        # normalize within wavefront sensor
        _, wfs = diffraction_functions.get_amplitude_mask_and_imagesize(self.N_interp, self.N_interp//3)
        wfs = np.expand_dims(wfs,0)
        wfs = np.expand_dims(wfs,-1)
        wfs = tf.constant(wfs,dtype=tf.float32)
        norm_factor = tf.reduce_max(wfs*tf.abs(self.field_cropped),keepdims=True,axis=[1,2])
        self.field_cropped = self.field_cropped / tf.complex(real=norm_factor,imag=tf.zeros_like(norm_factor))
        # return this as the field before wfs

        # propagator through wavefront sensor
        self.prop=self.propagate_tf.setup_graph_through_wfs(self.field_cropped)

        # to just multiply
        # self.prop = self.field_cropped * tf.complex(real=wfs,imag=tf.zeros_like(wfs))

        with tf.Session() as sess:
            # random numbers sbetween -6 and 6

            np.random.seed(12087)
            f={self.x: np.array([(12*np.random.rand(14))-6,
                                (12*np.random.rand(14))-6]),
                                self.scale:np.array([[1],[1]])
                                }

            # f={self.x: np.array([[10,0,0,0,0,5,0,0,0,0,0,0,0,0],
                                # [ 0,0,0,0,0,5,0,0,0,0,0,0,0,0]]),
                                # self.scale:np.array([[1],[1]])
                                # }
            out=sess.run(self.field_cropped,feed_dict=f)
            for i in [0,1]:
                fig,ax=plt.subplots(1,2)
                ax[0].imshow(np.real(out[i,:,:,0]),cmap='jet')
                ax[0].axhline(y=self.N_interp//2, color="black", alpha=0.5)
                ax[0].axvline(x=self.N_interp//2, color="black", alpha=0.5)
                ax[0].set_title("real")

                ax[1].imshow(np.imag(out[i,:,:,0]),cmap='jet')
                ax[1].axhline(y=self.N_interp//2, color="black", alpha=0.5)
                ax[1].axvline(x=self.N_interp//2, color="black", alpha=0.5)
                ax[1].set_title("imag")

            plt.figure()
            plt.title("0")
            plt.imshow(np.abs(out[0,:,:,0]),cmap='jet')
            plt.colorbar()

            plt.figure()
            plt.title("1")
            plt.imshow(np.abs(out[1,:,:,0]),cmap='jet')
            plt.colorbar()

            out=sess.run(self.prop,feed_dict=f)
            plt.figure()
            plt.title("0")
            plt.imshow(np.abs(out[0,:,:,0]),cmap='jet')
            plt.colorbar()

            plt.figure()
            plt.title("1")
            plt.imshow(np.abs(out[1,:,:,0]),cmap='jet')
            plt.colorbar()

            plt.show()
            exit()


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


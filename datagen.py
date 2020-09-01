import tensorflow as tf
import argparse
import tables
import numpy as np
import matplotlib.pyplot as plt
import math
import diffraction_functions


# def gaussian_propagate(zernike_polynom:tf.Tensor,scale:tf.Tensor)->tf.Tensor:
def plotsamples(beforewf:np.array,afterwf:np.array,z_coefs:np.array,scales:np.array,zernike_cvector:list,filesuffix:str):
    for i in range(np.shape(beforewf)[0]):
        _beforewf = np.squeeze(beforewf[i])
        _afterwf = np.squeeze(afterwf[i])
        _z_coefs = z_coefs[i]
        _scales = scales[i]

        fig=plt.figure(figsize=(10,10))
        fig.subplots_adjust(left=0.3)
        gs = fig.add_gridspec(2,2)

        fig.text(0.05,0.9,'Zernike Coefficients:',size=20,color='red')
        c_str=""
        for _c, _z in zip(_z_coefs,zernike_cvector):
            c_str += r"$Z^{"+str(_z.m)+"}_{"+str(_z.n)+"}$"
            c_str+="    "
            # c_str+=str(_z.m)+","
            # c_str+=str(_z.n)+"  "
            c_str+="%.2f"%_c+'\n'
        fig.text(0.03,0.85,c_str,ha='left',va='top',size=20)

        fig.text(0.05,0.15,'Scale:',size=20,color='red')
        fig.text(0.03,0.10,'S:'+"%.2f"%_scales[0],ha='left',va='top',size=20)

        ax=fig.add_subplot(gs[0,0])
        ax.set_title('intensity')
        im=ax.pcolormesh(np.abs(_beforewf)**2,cmap='jet')
        fig.colorbar(im,ax=ax)

        ax=fig.add_subplot(gs[0,1])
        ax.set_title('phase')
        phase = np.angle(_beforewf)
        phase[np.abs(_beforewf)**2 < 0.1*np.max(np.abs(_beforewf)**2)]=0.0
        im=ax.pcolormesh(phase,cmap='jet')
        fig.colorbar(im,ax=ax)

        ax=fig.add_subplot(gs[1,0])
        ax.set_title('after wfs')
        im=ax.pcolormesh(np.abs(_afterwf)**2,cmap='jet')
        fig.colorbar(im,ax=ax)

        ax=fig.add_subplot(gs[1,1])
        ax.set_title('diffraction pattern')
        difp = np.abs(np.fft.fftshift(np.fft.fft2(np.fft.fftshift(_afterwf))))**2
        difp *= 1/np.max(difp)
        im=ax.pcolormesh(difp,cmap='jet')
        fig.colorbar(im,ax=ax)
        fig.savefig(str(i)+filesuffix)



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
    def __init__(self,N_computational:int,N_interp:int):
        self.N_interp=N_interp
        self.N_computational=N_computational
        # generate zernike coefficients
        self.batch_size=4
        start_n=1
        max_n=4
        self.zernike_cvector = []

        for n in range(start_n,max_n+1):
            for m in range(n,-n-2,-2):
                self.zernike_cvector.append(Zernike_C(m,n))


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

    def buildgraph(self,x:tf.Tensor,scale:tf.Tensor)->(tf.Tensor,tf.Tensor):

        # generate polynomials
        zernikes=[]
        for i in range(len(self.zernike_cvector)):
            _z = self.zernike_cvector[i]
            zernikes.append(tf_make_zernike(_z.m,_z.n,self.N_computational,scale,x[:,i]))

        zernikes=tf.concat(zernikes,axis=1)
        zernike_polynom = tf.reduce_sum(zernikes,axis=1)

        # propagate through gaussian
        x = np.linspace(1,-1,self.N_computational).reshape(1,-1,1)
        y = np.linspace(1,-1,self.N_computational).reshape(1,1,-1)
        _scale = tf.expand_dims(scale,axis=-1)
        x = (1/_scale)*tf.constant(x,dtype=tf.float32)
        y = (1/_scale)*tf.constant(y,dtype=tf.float32)
        width = 0.05
        gaussian_amp = tf.exp(-(x**2)/(width**2))*tf.exp(-(y**2)/(width**2))

        field = tf.complex(real=gaussian_amp,imag=tf.zeros_like(gaussian_amp)) * tf.exp(tf.complex(real=tf.zeros_like(zernike_polynom),imag=zernike_polynom))

        # fft
        field = tf.expand_dims(field,axis=-1)
        field_ft=diffraction_functions.tf_fft2(field,dimmensions=[1,2])

        # crop and interpolate
        field_cropped = field_ft[:,(self.N_computational//2)-(self.N_interp//2):(self.N_computational//2)+(self.N_interp//2),(self.N_computational//2)-(self.N_interp//2):(self.N_computational//2)+(self.N_interp//2),:]

        z_center = field_cropped[:,self.N_interp//2,self.N_interp//2,0]
        z_center = tf.expand_dims(z_center,-1)
        z_center = tf.expand_dims(z_center,-1)
        z_center = tf.expand_dims(z_center,-1)
        field_cropped = field_cropped * tf.exp(tf.complex(real=0.0,imag=-1.0*tf.angle(z_center)))

        # normalize within wavefront sensor
        _, wfs = diffraction_functions.get_amplitude_mask_and_imagesize(self.N_interp, self.N_interp//3)
        wfs = np.expand_dims(wfs,0)
        wfs = np.expand_dims(wfs,-1)
        wfs = tf.constant(wfs,dtype=tf.float32)
        norm_factor = tf.reduce_max(wfs*tf.abs(field_cropped),keepdims=True,axis=[1,2])
        field_cropped = field_cropped / tf.complex(real=norm_factor,imag=tf.zeros_like(norm_factor))
        # return this as the field before wfs

        # propagator through wavefront sensor
        prop=self.propagate_tf.setup_graph_through_wfs(field_cropped)

        # to just multiply
        # self.prop = field_cropped * tf.complex(real=wfs,imag=tf.zeros_like(wfs))
        # TODO
        # check wavefront sensor image, up sampling problem
        return prop, field_cropped

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

def create_dataset(filename:str, coefficients:int):

    print("called create_dataset")
    print(filename)
    N = 128
    with tables.open_file(filename, "w") as hdf5file:

        # create array for the object
        hdf5file.create_earray(hdf5file.root, "object_real", tables.Float32Atom(), shape=(0,N*N))

        # create array for the object phase
        hdf5file.create_earray(hdf5file.root, "object_imag", tables.Float32Atom(), shape=(0,N*N))

        # create array for the image
        hdf5file.create_earray(hdf5file.root, "diffraction_noise", tables.Float32Atom(), shape=(0,N*N))

        # create array for the image
        hdf5file.create_earray(hdf5file.root, "diffraction_noisefree", tables.Float32Atom(), shape=(0,N*N))

        # scale
        hdf5file.create_earray(hdf5file.root, "scale", tables.Float32Atom(), shape=(0,1))

        # zernike coefficients
        hdf5file.create_earray(hdf5file.root, "coefficients", tables.Float32Atom(), shape=(0,coefficients))

        hdf5file.create_earray(hdf5file.root, "N", tables.Int32Atom(), shape=(0,1))

        hdf5file.close()

    with tables.open_file(filename, mode='a') as hd5file:
        # save the dimmensions of the data
        hd5file.root.N.append(np.array([[N]]))

def save_to_hdf5(filename:str, afterwf:np.array, beforewf:np.array, z_coefs:np.array, scales:np.array):
    with tables.open_file(filename, mode='a') as hd5file:
        for i in range(np.shape(beforewf)[0]):
            object_real = np.real(beforewf[i,:,:])
            object_imag = np.imag(beforewf[i,:,:])
            diffraction_pattern_noisefree = np.abs(np.fft.fftshift(np.fft.fft2(np.fft.fftshift(afterwf[i,:,:]))))**2
            _z_coefs = z_coefs[i,:]
            _scales = scales[i]


            # normalize
            diffraction_pattern_noisefree = diffraction_pattern_noisefree / np.max(diffraction_pattern_noisefree)
            diffraction_pattern_noisefree = diffraction_functions.center_image_at_centroid(diffraction_pattern_noisefree)
            hd5file.root.object_real.append(object_real.reshape(1,-1))
            hd5file.root.object_imag.append(object_imag.reshape(1,-1))
            hd5file.root.diffraction_noisefree.append(diffraction_pattern_noisefree.reshape(1,-1))
            hd5file.root.coefficients.append(_z_coefs.reshape(1,-1))
            hd5file.root.scale.append(_scales.reshape(1,-1))

        print("calling flush")
        hd5file.flush()

if __name__ == "__main__":

    parser=argparse.ArgumentParser()
    parser.add_argument('--count',type=int)
    parser.add_argument('--seed',type=int)
    parser.add_argument('--name',type=str)
    parser.add_argument('--batch_size',type=int)
    args=parser.parse_args()
    if args.count % args.batch_size != 0:
        raise ValueError('batch size and count divide with remainder')

    datagenerator = DataGenerator(1024,128)

    x = tf.placeholder(tf.float32, shape=[None, len(datagenerator.zernike_cvector)])
    scale = tf.placeholder(tf.float32, shape=[None,1])
    afterwf,beforewf=datagenerator.buildgraph(x,scale)

    create_dataset(filename=args.name,coefficients=len(datagenerator.zernike_cvector))
    with tf.Session() as sess:
        np.random.seed(args.seed)
        _count = 0
        while _count<args.count:
            print("_count =>", _count)
            # make random numbers

            # for the zernike coefs
            n_z_coefs=len(datagenerator.zernike_cvector)* args.batch_size
            # for the scales
            n_scales=args.batch_size

            z_coefs = 12*(np.random.rand(n_z_coefs)-0.5)
            z_coefs=z_coefs.reshape(args.batch_size,-1)
            scales = 1+1*(np.random.rand(n_scales)-0.5)
            scales = scales.reshape(args.batch_size,1)
            # z_coefs[:,0:3]=0
            # z_coefs[:,9:]=0 # doesnt work
            # z_coefs[:,8:]=0 # works
            f={x: z_coefs,
               scale:scales
                                }
            _afterwf=sess.run(afterwf,feed_dict=f)
            _beforewf=sess.run(beforewf,feed_dict=f)
            save_to_hdf5(
                    args.name,
                    np.squeeze(_afterwf),
                    np.squeeze(_beforewf),
                    np.squeeze(z_coefs),
                    np.squeeze(scales)
                    )
            # plot data
            # for i in range(2):
                # fig,ax=plt.subplots(1,2,figsize=(10,5))
                # im=ax[0].imshow(np.abs(_beforewf[i,:,:,0])**2,cmap='jet')
                # ax[0].set_title("intensity")
                # fig.colorbar(im,ax=ax[0])
                # im=ax[1].imshow(np.angle(_beforewf[i,:,:,0]),cmap='jet')
                # ax[1].set_title("angle")
                # fig.colorbar(im,ax=ax[1])
            # plt.show()
            _count += args.batch_size

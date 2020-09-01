import tables
import pickle
import numpy as np
import datagen
import diffraction_net
import matplotlib.pyplot as plt
import tensorflow as tf

class Sample():
    index=None
    beforewf=None
    afterwf=None
    z_coefs=None
    scales=None
    def __init__(self,beforewf,afterwf,z_coefs,scales,index):
        self.index=index
        self.beforewf=beforewf
        self.afterwf=afterwf
        self.z_coefs=z_coefs
        self.scales=scales

if __name__=="__main__":
    getdata = diffraction_net.GetData(10,"_allwithlin_andscale_nrtest1_fixeccostf3")

    datagenerator = datagen.DataGenerator(1024,128)
    x = tf.placeholder(tf.float32, shape=[None, len(datagenerator.zernike_cvector)])
    scale = tf.placeholder(tf.float32, shape=[None,1])
    afterwf,beforewf=datagenerator.buildgraph(x,scale)
    with tf.Session() as sess:

        A=None # maximum value
        sample1=None
        sample2=None
        for i in range(5):
            i_diffraction=getdata.hdf5_file_train.root.diffraction_noisefree[i].reshape(1,getdata.N,getdata.N,1)
            i_z_coefs=getdata.hdf5_file_train.root.coefficients[i].reshape(1,-1)
            i_scales=getdata.hdf5_file_train.root.scale[i].reshape(1,-1)
            f={x: i_z_coefs,
               scale:i_scales}
            i__afterwf=sess.run(afterwf,feed_dict=f)
            i__beforewf=sess.run(beforewf,feed_dict=f)

            for j in range(getdata.samples):
                print("searching:",i,j)
                if i == j:
                    continue

                j_diffraction=getdata.hdf5_file_train.root.diffraction_noisefree[j].reshape(1,getdata.N,getdata.N,1)
                j_z_coefs=getdata.hdf5_file_train.root.coefficients[j].reshape(1,-1)
                j_scales=getdata.hdf5_file_train.root.scale[j].reshape(1,-1)
                f={x: j_z_coefs,
                   scale:j_scales}
                j__afterwf=sess.run(afterwf,feed_dict=f)
                j__beforewf=sess.run(beforewf,feed_dict=f)

                # calculate difference in diffraction patterns
                diffraction_func_diff = np.sqrt(np.square(j_diffraction.reshape(-1)-i_diffraction.reshape(-1)).mean())
                real_diff = np.sqrt(np.square(np.real(j__beforewf).reshape(-1)-np.real(i__beforewf).reshape(-1)).mean())
                imag_diff = np.sqrt(np.square(np.imag(j__beforewf).reshape(-1)-np.imag(i__beforewf).reshape(-1)).mean())

                # _A = -4*diffraction_func_diff + real_diff + imag_diff
                _A = -4*diffraction_func_diff
                # find maximum value for A
                if not A or _A > A:
                    A = _A
                    print("new max A:",A)
                    sample1=Sample(i__beforewf,i__afterwf,i_z_coefs,i_scales,i)
                    sample2=Sample(j__beforewf,j__afterwf,j_z_coefs,j_scales,j)

                    # datagen.plotsamples(sample1.beforewf,sample1.afterwf,sample1.z_coefs,sample1.scales,datagenerator.zernike_cvector,'test')
                    # datagen.plotsamples(sample2.beforewf,sample2.afterwf,sample2.z_coefs,sample2.scales,datagenerator.zernike_cvector,'test')
                    with open("sample1.p","wb") as file:
                        pickle.dump(sample1,file)
                    with open("sample2.p","wb") as file:
                        pickle.dump(sample2,file)

            # datagen.plotsamples(_beforewf,_afterwf,z_coefs,scales,datagenerator.zernike_cvector,'test')
            # find most similar diffraction pattern

    datagen.plotsamples(sample1.beforewf,sample1.afterwf,sample1.z_coefs,sample1.scales,datagenerator.zernike_cvector,'test')
    datagen.plotsamples(sample2.beforewf,sample2.afterwf,sample2.z_coefs,sample2.scales,datagenerator.zernike_cvector,'test')
    plt.show()

import numpy as np
from scipy.ndimage import shift as sc_im_shift
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import tables
import os
import diffraction_functions
import argparse
import datagen

class CameraNoise():
    def __init__(self,imagefile):
        print("init camera noise")
        print("imagefile =>", imagefile)
        self.imagefile=imagefile
        self.im = Image.open(self.imagefile)
        self.im=self.im.convert("L")
        self.im=np.array(self.im)

        # flatten the array
        self.distribution=self.im.reshape(-1)
        avg=np.average(self.distribution)
        print("distribution avergae:"+str(avg))

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--infile',type=str)
    parser.add_argument('--outfile',type=str)
    parser.add_argument('--peakcount',type=int)
    parser.add_argument('--cameraimage',type=str)
    args=parser.parse_args()

    ocameraNoise=CameraNoise(args.cameraimage)

    # print("parser.INFILE =>", args.infile)
    print("args.infile =>", args.infile)
    print("args.outfile =>", args.outfile)
    print("args.peakcount =>", args.peakcount)
    print("args.cameraimage =>", args.cameraimage)

    # create new hdf5 file
    with tables.open_file(args.infile,mode="r") as hd5file:
        # get number of coefficients
        n_z_coefs = len(hd5file.root.coefficients[0])
        datagen.create_dataset(args.outfile,coefficients=n_z_coefs)
        with tables.open_file(args.outfile,mode="a") as newhd5file:

            N = hd5file.root.N[0,0]
            samples = hd5file.root.object_real.shape[0]
            print("samples =>", samples)
            for _i in range(samples):

                if _i % 100 == 0:
                    print("add noise to sample:"+str(_i))

                object_real=hd5file.root.object_real[_i, :].reshape(N,N)
                object_imag=hd5file.root.object_imag[_i, :].reshape(N,N)
                diffraction=hd5file.root.diffraction_noisefree[_i, :].reshape(N,N)
                _z_coefs = hd5file.root.coefficients[_i, :]
                _scales = hd5file.root.scale[_i,:]


                if args.peakcount>0:
                    # apply poisson noise
                    peak_signal_counts=args.peakcount
                    scalar=peak_signal_counts/np.max(diffraction)
                    diffraction_pattern_with_noise_poisson=diffraction*scalar
                    diffraction_pattern_with_noise_poisson=np.random.poisson(diffraction_pattern_with_noise_poisson)

                    # draw from random sample
                    # apply camera noise
                    total_sim_size=diffraction.shape[0]*diffraction.shape[1]
                    camera_noise=np.random.choice(ocameraNoise.distribution,size=total_sim_size)
                    camera_noise=camera_noise.reshape(diffraction.shape)

                    diffraction_pattern_with_noise_poisson_and_camera=diffraction_pattern_with_noise_poisson+camera_noise
                    # normalize
                    diffraction_pattern_with_noise_poisson_and_camera=diffraction_pattern_with_noise_poisson_and_camera/np.max(diffraction_pattern_with_noise_poisson_and_camera)

                    # apply random position shift by sub pixel
                    dis_y, dis_x = np.random.rand()*2 -1, np.random.rand()*2 -1
                    diffraction_pattern_with_noise_poisson_and_camera = sc_im_shift(diffraction_pattern_with_noise_poisson_and_camera,(dis_y,dis_x))
                    diffraction_pattern_with_noise_poisson_and_camera = diffraction_functions.center_image_at_centroid(diffraction_pattern_with_noise_poisson_and_camera)

                    newhd5file.root.object_real.append(object_real.reshape(1,-1))
                    newhd5file.root.object_imag.append(object_imag.reshape(1,-1))
                    newhd5file.root.diffraction_noise.append(diffraction_pattern_with_noise_poisson_and_camera.reshape(1,-1))
                    newhd5file.root.diffraction_noisefree.append(diffraction.reshape(1,-1))
                    newhd5file.root.coefficients.append(_z_coefs.reshape(1,-1))
                    newhd5file.root.scale.append(_scales.reshape(1,-1))

                else:
                    newhd5file.root.object_real.append(object_real.reshape(1,-1))
                    newhd5file.root.object_imag.append(object_imag.reshape(1,-1))
                    newhd5file.root.diffraction_noisefree.append(diffraction.reshape(1,-1))

                    # apply random position shift by sub pixel
                    dis_y, dis_x = np.random.rand()*2 -1, np.random.rand()*2 -1
                    diffraction = sc_im_shift(diffraction,(dis_y,dis_x))
                    diffraction = diffraction_functions.center_image_at_centroid(diffraction)

                    newhd5file.root.diffraction_noise.append(diffraction.reshape(1,-1))
                    newhd5file.root.coefficients.append(_z_coefs.reshape(1,-1))
                    newhd5file.root.scale.append(_scales.reshape(1,-1))


                # diffraction=diffraction_pattern_with_noise_poisson_and_camera

                # plt.figure()
                # plt.pcolormesh(diffraction_pattern_with_noise_poisson_and_camera)
                # plt.title("diffraction_pattern_with_noise_poisson_and_camera")

                # plt.figure()
                # plt.pcolormesh(diffraction)
                # plt.title("diffraction")

                # plt.show()
                # # exit()



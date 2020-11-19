import diffraction_net
import scipy.io
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
import time
import diffraction_functions
import glob
import shutil
import imageio
from numpy import unravel_index

class NetworkRetrieval(diffraction_net.DiffractionNet):
    def __init__(self, name):

        self.name = name
        # copy the weights and create a new network
        for file in glob.glob(r'./models/{}.ckpt.*'.format(self.name)):
            file_newname = file.replace(self.name, self.name+'_retrieval')
            shutil.copy(file, file_newname)

        for file in glob.glob(r'./models/{}.p'.format(self.name)):
            file_newname = file.replace(self.name, self.name+'_retrieval')

        shutil.copy(r'./models/{}.p'.format(self.name),
                r'./models/{}.p'.format(self.name+'_retrieval'))

        diffraction_net.DiffractionNet.__init__(self, self.name+'_retrieval')

    def retrieve_experimental(self):
        # data = self.get_data.evaluate_on_train_data(n_samples=10)
        # with open("training_data_samples.p", "wb") as file:
            # pickle.dump(data, file)
        # exit()
        with open("training_data_samples.p", "rb") as file:
            data = pickle.load(file)


        object_real_samples = data["object_real_samples"].reshape(
                -1,self.get_data.N, self.get_data.N, 1)
        object_imag_samples = data["object_imag_samples"].reshape(
                -1,self.get_data.N, self.get_data.N, 1)
        diffraction_samples = data["diffraction_samples"].reshape(
                -1,self.get_data.N, self.get_data.N, 1)

        # get the output
        real_output = self.sess.run(self.nn_nodes["real_out"], feed_dict={self.x:diffraction_samples})
        imag_output = self.sess.run(self.nn_nodes["imag_out"], feed_dict={self.x:diffraction_samples})
        tf_reconstructed_diff = self.sess.run(
                self.nn_nodes["recons_diffraction_pattern"], feed_dict={self.x:diffraction_samples})

        index = 1
        plotretrieval(
                plot_title = "sample "+str(index),
                object_real_samples = object_real_samples[index,:,:,0],
                object_imag_samples = object_imag_samples[index,:,:,0],
                diffraction_samples = diffraction_samples[index,:,:,0],
                real_output = real_output[index,:,:,0],
                imag_output = imag_output[index,:,:,0],
                tf_reconstructed_diff = tf_reconstructed_diff[index,:,:,0]
                )

        index = 2
        plotretrieval(
                plot_title = "sample "+str(index),
                object_real_samples = object_real_samples[index,:,:,0],
                object_imag_samples = object_imag_samples[index,:,:,0],
                diffraction_samples = diffraction_samples[index,:,:,0],
                real_output = real_output[index,:,:,0],
                imag_output = imag_output[index,:,:,0],
                tf_reconstructed_diff = tf_reconstructed_diff[index,:,:,0]
                )

        measured_pattern = self.get_and_format_experimental_trace()

        diffraction_functions.plot_image_show_centroid_distance(
                np.squeeze(measured_pattern),
                "measured_pattern",
                10)

        index = 1
        diffraction_functions.plot_image_show_centroid_distance(
                diffraction_samples[index,:,:,0],
                "diffraction_samples[index,:,:,0]",
                11)

        # retrieve the experimental diffraction pattern
        real_output = self.sess.run(self.nn_nodes["real_out"], feed_dict={self.x:measured_pattern})
        imag_output = self.sess.run(self.nn_nodes["imag_out"], feed_dict={self.x:measured_pattern})
        tf_reconstructed_diff = self.sess.run(self.nn_nodes["recons_diffraction_pattern"], feed_dict={self.x:measured_pattern})


        # "experimental data:"
        plotretrieval(
                plot_title = "experimental data",
                object_real_samples = None,
                object_imag_samples = None,
                diffraction_samples = measured_pattern[0,:,:,0],
                real_output = real_output[0,:,:,0],
                imag_output = imag_output[0,:,:,0],
                tf_reconstructed_diff = tf_reconstructed_diff[0,:,:,0]
                )

        plt.show()

    def get_and_format_experimental_trace(self, transform):
        # TODO double check this works
        return diffraction_functions.get_and_format_experimental_trace(self.N, transform)

    def unsupervised_retrieval(self, transform, iterations, plotting=True):
        """
        transform: dict{}
        """
        measured_pattern = self.get_and_format_experimental_trace(transform)

        # retrieve the experimental diffraction pattern
        real_output = self.sess.run(self.nn_nodes["real_out"], feed_dict={self.x:measured_pattern})
        imag_output = self.sess.run(self.nn_nodes["imag_out"], feed_dict={self.x:measured_pattern})
        tf_reconstructed_diff = self.sess.run(self.nn_nodes["recons_diffraction_pattern"], feed_dict={self.x:measured_pattern})

        if plotting == True:
            plt.ion()
            fig = plt.figure(1, figsize=(8,10))
            gs = fig.add_gridspec(3,2)
            measured_pattern_ax = fig.add_subplot(gs[0,0])
            reconstructed_ax = fig.add_subplot(gs[0,1])
            retrieved_real_ax = fig.add_subplot(gs[1,1])
            retrieved_imag_ax = fig.add_subplot(gs[2,1])

            # set text to show epoch
            epoch_text = fig.text(0.4, 0.5,"epoch: {}".format(0), fontsize=10, ha='center', backgroundcolor="yellow")


            measured_pattern_im = measured_pattern_ax.imshow(measured_pattern[0,:,:,0])
            measured_pattern_text = measured_pattern_ax.text(0.3, 0.9,"measured diffraction pattern", fontsize=10, ha='center', transform=measured_pattern_ax.transAxes, backgroundcolor="yellow")

            # reconstructed diffraction pattern
            reconstructed_im = reconstructed_ax.imshow(tf_reconstructed_diff[0,:,:,0])
            reconstructed_text = reconstructed_ax.text(0.3, 0.9,"reconstructed diffraction pattern", fontsize=10, ha='center', transform=reconstructed_ax.transAxes, backgroundcolor="yellow")

            # retrieved real object
            retrieved_real_im = retrieved_real_ax.imshow(real_output[0,:,:,0])
            retrieved_real_text = retrieved_real_ax.text(0.3, 0.9,"retrieved real", fontsize=10, ha='center', transform=retrieved_real_ax.transAxes, backgroundcolor="yellow")

            # retrieved imag object
            retrieved_imag_im = retrieved_imag_ax.imshow(imag_output[0,:,:,0])
            retrieved_imag_text = retrieved_imag_ax.text(0.3, 0.9,"retrieved imag", fontsize=10, ha='center', transform=retrieved_imag_ax.transAxes, backgroundcolor="yellow")

        retrieved_obj = {}
        retrieved_obj["measured_pattern"] = measured_pattern
        for i in range(iterations):
            # run the training for minimizing the retreival error
            # TODO
            if i % 5 == 0:
                print("epoch: "+str(i))

                if plotting==True:
                    epoch_text.set_text("epoch: {}".format(i))
                    # retrieve the experimental diffraction pattern
                    retrieved_obj["real_output"] = self.sess.run(self.nn_nodes["real_out"], feed_dict={self.x:measured_pattern})
                    retrieved_obj["imag_output"] = self.sess.run(self.nn_nodes["imag_out"], feed_dict={self.x:measured_pattern})
                    retrieved_obj["tf_reconstructed_diff"] = self.sess.run(self.nn_nodes["recons_diffraction_pattern"], feed_dict={self.x:measured_pattern})
                    reconstructed_im.set_data(retrieved_obj["tf_reconstructed_diff"][0,:,:,0])
                    retrieved_real_im.set_data(retrieved_obj["imag_output"][0,:,:,0])
                    retrieved_imag_im.set_data(retrieved_obj["real_output"][0,:,:,0])

                    # Used to return the plot as an image rray
                    plt.pause(0.1)

            # run unsupervsied training
            self.sess.run(self.nn_nodes["u_train"], feed_dict={self.x:measured_pattern, self.s_LR:0.0001})

        retrieved_obj["real_output"] = self.sess.run(self.nn_nodes["real_out"], feed_dict={self.x:measured_pattern})
        retrieved_obj["imag_output"] = self.sess.run(self.nn_nodes["imag_out"], feed_dict={self.x:measured_pattern})
        retrieved_obj["tf_reconstructed_diff"] = self.sess.run(self.nn_nodes["recons_diffraction_pattern"], feed_dict={self.x:measured_pattern})

        return retrieved_obj

    def close(self):
        self.sess.close()


def plotretrieval(plot_title, object_real_samples, object_imag_samples, diffraction_samples,
                    real_output, imag_output, tf_reconstructed_diff):

    # plot the output of the network
    axes_obj = diffraction_net.PlotAxes(plot_title)

    im = axes_obj.diffraction_samples.pcolormesh(diffraction_samples)
    axes_obj.fig.colorbar(im, ax=axes_obj.diffraction_samples)

    if object_real_samples is not None:
        im = axes_obj.object_real_samples.pcolormesh(object_real_samples)
        axes_obj.fig.colorbar(im, ax=axes_obj.object_real_samples)

    if object_imag_samples is not None:
        im = axes_obj.object_imag_samples.pcolormesh(object_imag_samples)
        axes_obj.fig.colorbar(im, ax=axes_obj.object_imag_samples)

    im = axes_obj.real_output.pcolormesh(real_output)
    axes_obj.fig.colorbar(im, ax=axes_obj.real_output)

    im = axes_obj.imag_output.pcolormesh(imag_output)
    axes_obj.fig.colorbar(im, ax=axes_obj.imag_output)

    # tensorflow reconstructed diffraction pattern
    im = axes_obj.tf_reconstructed_diff.pcolormesh(tf_reconstructed_diff)
    axes_obj.fig.colorbar(im, ax=axes_obj.tf_reconstructed_diff)


def test_various_scales(scales, orientation, iterations):

    save_folder = "flip_"+str(orientation)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    for scale in scales:
        print("scale:"+str(scale))
        network_retrieval = NetworkRetrieval("IGLUY_constrain_output_with_mask_u")
        transform = {}
        transform["scale"] = scale
        transform["flip"] = orientation
        retrieved_obj = network_retrieval.unsupervised_retrieval(transform, iterations, plotting=False)

        # save the retrieval
        title = "orientation change:"+str(orientation)+"    "+"scale:"+str(scale)
        fig = diffraction_functions.plot_amplitude_phase_meas_retreival(retrieved_obj, title)

        filename = str(orientation)+"_"+str(scale).replace(".", "_")+".png"
        fig.savefig(os.path.join(save_folder,filename))

        amplitude_mask = network_retrieval.amplitude_mask
        network_retrieval.close()
        del network_retrieval
        tf.reset_default_graph()

        # retrieve the diffraction pattern with matlab code
        retrieved_obj = matlab_cdi_retrieval(np.squeeze(retrieved_obj["measured_pattern"]), np.squeeze(amplitude_mask))
        # save the retrieval
        title = "MATLAB CDI orientation change:"+str(orientation)+"    "+"scale:"+str(scale)
        fig = diffraction_functions.plot_amplitude_phase_meas_retreival(retrieved_obj, title)
        filename = "MATLABCDI_"+str(orientation)+"_"+str(scale).replace(".", "_")+".png"
        fig.savefig(os.path.join(save_folder,filename))

def matlab_cdi_retrieval(diffraction_pattern, support):

    # move to matlab cdi folder
    start_dir = os.getcwd()
    os.chdir("matlab_cdi")

    randomid_num = np.random.randint(10,size=10)
    randomid = ""
    for r in randomid_num:
        randomid += str(r)

    diffraction_pattern_file = randomid + "_diffraction.mat"
    support_file = randomid + "_support.mat"

    retrieved_obj_file = randomid + "_retrieved_obj.mat"
    reconstructed_file = randomid + "_reconstructed.mat"

    scipy.io.savemat(support_file, {'support':support})
    scipy.io.savemat(diffraction_pattern_file, {'diffraction':diffraction_pattern})

    # matlab load file
    with open("loaddata.m", "w") as file:
        file.write("function [diffraction_pattern_file, support_file, retrieved_obj_file, reconstructed_file] = loaddata()\n")
        file.write("diffraction_pattern_file = '{}';\n".format(diffraction_pattern_file))
        file.write("support_file = '{}';\n".format(support_file))
        file.write("retrieved_obj_file = '{}';\n".format(retrieved_obj_file))
        file.write("reconstructed_file = '{}';\n".format(reconstructed_file))
        file.flush()
    os.system('matlab -nodesktop -r seeded_run_CDI_noprocessing')

    # load the results from matlab run
    rec_object = scipy.io.loadmat(retrieved_obj_file)['rec_object']
    recon_diffracted = scipy.io.loadmat(reconstructed_file)['recon_diffracted']

    # go back to starting dir
    os.chdir(start_dir)

    retrieved_obj = {}
    retrieved_obj["measured_pattern"] = diffraction_pattern
    retrieved_obj["tf_reconstructed_diff"] = recon_diffracted
    retrieved_obj["real_output"] = np.real(rec_object)
    retrieved_obj["imag_output"] = np.imag(rec_object)
    return retrieved_obj



if __name__ == "__main__":


    # scales = np.arange(0.8, 1.2, 0.01)
    scales = np.array([0.9, 1.0, 1.1])
    test_various_scales(scales, None, 300)
    test_various_scales(scales, "lr", 300)
    test_various_scales(scales, "ud", 300)
    test_various_scales(scales, "lrud", 300)



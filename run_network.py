import diffraction_net
import numpy as np
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
import time
import diffraction_functions
import glob
import shutil
import imageio

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
        # get the measured trace
        # open the object with known dimmensions
        obj_calculated_measured_axes, _ = diffraction_functions.get_amplitude_mask_and_imagesize(self.get_data.N, int(self.get_data.N/2))
        diffraction_calculated_measured_axes, measured_pattern = diffraction_functions.get_measured_diffraction_pattern_grid()

        measured_pattern = measured_pattern.astype(np.float64)
        # measured_pattern = measured_pattern.T


        df_ratio = diffraction_calculated_measured_axes['diffraction_plane']['df'] / obj_calculated_measured_axes['diffraction_plane']['df']
        # multiply by scale adjustment
        df_ratio *= transform["scale"]

        measured_pattern = diffraction_functions.format_experimental_trace(
                N=self.get_data.N,
                df_ratio=df_ratio,
                measured_diffraction_pattern=measured_pattern,
                rotation_angle=3,
                trim=1) # if transposed (measured_pattern.T) , flip the rotation
                # use 30 to block outer maxima

        # diffraction_functions.plot_image_show_centroid_distance(
                # measured_pattern,
                # "before flip",
                # 10)

        if transform["flip"] == "lr":
            measured_pattern = np.flip(measured_pattern, axis=1)

        elif transform["flip"] == "ud":
            measured_pattern = np.flip(measured_pattern, axis=0)

        elif transform["flip"] == "lrud":
            measured_pattern = np.flip(measured_pattern, axis=0)
            measured_pattern = np.flip(measured_pattern, axis=1)

        elif transform["flip"] == None:
            pass

        else:
            raise ValueError("invalid flip specified")

        # diffraction_functions.plot_image_show_centroid_distance(
                # measured_pattern,
                # "after flip",
                # 11)

        measured_pattern = diffraction_functions.center_image_at_centroid(measured_pattern)

        # diffraction_functions.plot_image_show_centroid_distance(
                # measured_pattern,
                # "after flip,  center_image_at_centroid",
                # 12)

        measured_pattern = np.expand_dims(measured_pattern, axis=0)
        measured_pattern = np.expand_dims(measured_pattern, axis=-1)
        # print("np.max(measured_pattern[0,:,:,0]) =>", np.max(measured_pattern[0,:,:,0]))
        measured_pattern *= (1/np.max(measured_pattern))

        return measured_pattern

    def unsupervised_retrieval(self, transform, iterations):
        """
        transform: dict{}
        """
        measured_pattern = self.get_and_format_experimental_trace(transform)

        # retrieve the experimental diffraction pattern
        real_output = self.sess.run(self.nn_nodes["real_out"], feed_dict={self.x:measured_pattern})
        imag_output = self.sess.run(self.nn_nodes["imag_out"], feed_dict={self.x:measured_pattern})
        tf_reconstructed_diff = self.sess.run(self.nn_nodes["recons_diffraction_pattern"], feed_dict={self.x:measured_pattern})

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
                epoch_text.set_text("epoch: {}".format(i))
                # retrieve the experimental diffraction pattern
                retrieved_obj["real_output"] = self.sess.run(self.nn_nodes["real_out"], feed_dict={self.x:measured_pattern})
                retrieved_obj["imag_output"] = self.sess.run(self.nn_nodes["imag_out"], feed_dict={self.x:measured_pattern})
                retrieved_obj["tf_reconstructed_diff"] = self.sess.run(self.nn_nodes["recons_diffraction_pattern"], feed_dict={self.x:measured_pattern})
                reconstructed_im.set_data(retrieved_obj["tf_reconstructed_diff"][0,:,:,0])
                retrieved_real_im.set_data(retrieved_obj["imag_output"][0,:,:,0])
                retrieved_imag_im.set_data(retrieved_obj["real_output"][0,:,:,0])
                print("epoch: "+str(i))

                # Used to return the plot as an image rray
                plt.pause(0.1)

            # run unsupervsied training
            self.sess.run(self.nn_nodes["u_train"], feed_dict={self.x:measured_pattern, self.s_LR:0.0001})

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

def plot_amplitude_phase_meas_retreival(retrieved_obj, title):

    # print(retrieved_obj.keys())

    fig = plt.figure(figsize=(10,10))
    gs = fig.add_gridspec(3,2)
    fig.text(0.5, 0.95, title, ha="center", size=30)

    axes = {}
    axes["measured"] = fig.add_subplot(gs[0,0])
    axes["reconstructed"] = fig.add_subplot(gs[0,1])

    axes["real"] = fig.add_subplot(gs[1,0])
    axes["imag"] = fig.add_subplot(gs[1,1])

    axes["intensity"] = fig.add_subplot(gs[2,0])
    axes["phase"] = fig.add_subplot(gs[2,1])

    # calculate the intensity
    complex_obj = np.squeeze(retrieved_obj["real_output"]) + 1j * np.squeeze(retrieved_obj["imag_output"])

    I = np.abs(complex_obj)**2

    # calculate the phase
    obj_phase = np.angle(complex_obj)

    im = axes["phase"].pcolormesh(obj_phase)
    axes["phase"].text(0.2, 0.9,"phase(retrieved)", fontsize=10, ha='center', transform=axes["phase"].transAxes, backgroundcolor="cyan")
    fig.colorbar(im, ax=axes["phase"])

    im = axes["intensity"].pcolormesh(I)
    axes["intensity"].text(0.2, 0.9,"intensity(retrieved)", fontsize=10, ha='center', transform=axes["intensity"].transAxes, backgroundcolor="cyan")
    fig.colorbar(im, ax=axes["intensity"])

    im = axes["measured"].pcolormesh(np.squeeze(retrieved_obj["measured_pattern"]))
    axes["measured"].text(0.2, 0.9,"measured", fontsize=10, ha='center', transform=axes["measured"].transAxes, backgroundcolor="cyan")
    fig.colorbar(im, ax=axes["measured"])

    im = axes["reconstructed"].pcolormesh(np.squeeze(retrieved_obj["tf_reconstructed_diff"]))
    axes["reconstructed"].text(0.2, 0.9,"reconstructed", fontsize=10, ha='center', transform=axes["reconstructed"].transAxes, backgroundcolor="cyan")
    fig.colorbar(im, ax=axes["reconstructed"])

    im = axes["real"].pcolormesh(np.squeeze(retrieved_obj["real_output"]))
    axes["real"].text(0.2, 0.9,"real(retrieved)", fontsize=10, ha='center', transform=axes["real"].transAxes, backgroundcolor="cyan")
    fig.colorbar(im, ax=axes["real"])

    im = axes["imag"].pcolormesh(np.squeeze(retrieved_obj["imag_output"]))
    axes["imag"].text(0.2, 0.9,"imag(retrieved)", fontsize=10, ha='center', transform=axes["imag"].transAxes, backgroundcolor="cyan")
    fig.colorbar(im, ax=axes["imag"])

    return fig



if __name__ == "__main__":
    network_retrieval = NetworkRetrieval("IGLUY_constrain_output_with_mask_u")
    # network_retrieval.retrieve_experimental()
    transform = {}
    # between 0.4 and 1.4
    transform["scale"] = 1.0
    # transform["flip"] = "lr"
    # transform["flip"] = "ud"
    transform["flip"] = "lrud"
    # transform["flip"] = None
    retrieved_obj = network_retrieval.unsupervised_retrieval(transform, iterations=20)
    # del network_retrieval
    network_retrieval.close()
    del network_retrieval
    tf.reset_default_graph()


    network_retrieval = NetworkRetrieval("IGLUY_constrain_output_with_mask_u")
    # network_retrieval.retrieve_experimental()
    transform = {}
    # between 0.4 and 1.4
    transform["scale"] = 1.0
    # transform["flip"] = "lr"
    transform["flip"] = None
    # transform["flip"] = "ud"
    # transform["flip"] = "lrud"
    retrieved_obj = network_retrieval.unsupervised_retrieval(transform, iterations=20)
    # del network_retrieval
    network_retrieval.close()
    del network_retrieval

    # # with open("retrieved_obj.p", "wb") as file:
        # # pickle.dump(retrieved_obj, file)
    # # exit()
    # with open("retrieved_obj.p", "rb") as file:
        # retrieved_obj = pickle.load(file)

    title = "lr up sclae: 34"
    fig = plot_amplitude_phase_meas_retreival(retrieved_obj, title)
    fig.savefig("fig.png")

    plt.ioff()
    plt.show()



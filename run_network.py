import diffraction_net
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
import diffraction_functions

class NetworkRetrieval(diffraction_net.DiffractionNet):
    def __init__(self, name):
        diffraction_net.DiffractionNet.__init__(self, name)

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

        # get the measured trace
        # open the object with known dimmensions
        obj_calculated_measured_axes, _ = diffraction_functions.get_amplitude_mask_and_imagesize(self.get_data.N, int(self.get_data.N/2))
        diffraction_calculated_measured_axes, measured_pattern = diffraction_functions.get_measured_diffraction_pattern_grid()

        measured_pattern = measured_pattern.astype(np.float64)
        measured_pattern = measured_pattern.T


        df_ratio = diffraction_calculated_measured_axes['diffraction_plane']['df'] / obj_calculated_measured_axes['diffraction_plane']['df']

        measured_pattern = diffraction_functions.format_experimental_trace(
                N=self.get_data.N,
                df_ratio=df_ratio,
                measured_diffraction_pattern=measured_pattern,
                rotation_angle=-3,
                trim=1) # if transposed (measured_pattern.T) , flip the rotation
                # use 30 to block outer maxima

        diffraction_functions.plot_image_show_centroid_distance(
                measured_pattern,
                "measured_pattern",
                10)

        index = 1
        diffraction_functions.plot_image_show_centroid_distance(
                diffraction_samples[index,:,:,0],
                "diffraction_samples[index,:,:,0]",
                11)

        measured_pattern = np.expand_dims(measured_pattern, axis=0)
        measured_pattern = np.expand_dims(measured_pattern, axis=-1)
        # print("np.max(measured_pattern[0,:,:,0]) =>", np.max(measured_pattern[0,:,:,0]))
        measured_pattern *= (1/np.max(measured_pattern))
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


if __name__ == "__main__":
    network_retrieval = NetworkRetrieval("IUBL_centered_at_centroid1")


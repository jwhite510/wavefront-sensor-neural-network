import diffraction_net
import numpy as np
import matplotlib.pyplot as plt

class NetworkRetrieval(diffraction_net.DiffractionNet):
    def __init__(self, name):
        diffraction_net.DiffractionNet.__init__(self, name)

        data = self.get_data.evaluate_on_train_data(n_samples=50)

        object_real_samples = data["object_real_samples"].reshape(-1,self.get_data.N, self.get_data.N, 1)
        object_imag_samples = data["object_imag_samples"].reshape(-1,self.get_data.N, self.get_data.N, 1)
        diffraction_samples = data["diffraction_samples"].reshape(-1,self.get_data.N, self.get_data.N, 1)

        # get the output
        real_output = self.sess.run(self.nn_nodes["real_out"], feed_dict={self.x:diffraction_samples})
        imag_output = self.sess.run(self.nn_nodes["imag_out"], feed_dict={self.x:diffraction_samples})
        tf_reconstructed_diff = self.sess.run(self.nn_nodes["recons_diffraction_pattern"], feed_dict={self.x:diffraction_samples})

        print("np.shape(object_real_samples) => ",np.shape(object_real_samples))
        print("np.shape(object_imag_samples) => ",np.shape(object_imag_samples))
        print("np.shape(diffraction_samples) => ",np.shape(diffraction_samples))
        print("np.shape(real_output) => ",np.shape(real_output))
        print("np.shape(imag_output) => ",np.shape(imag_output))
        print("np.shape(tf_reconstructed_diff) => ",np.shape(tf_reconstructed_diff))

        index = 1

        plt.figure()
        plt.imshow(np.squeeze(diffraction_samples[index]))
        plt.savefig("./1.png")

        plt.figure()
        plt.imshow(np.squeeze(tf_reconstructed_diff[index]))
        plt.savefig("./2.png")


if __name__ == "__main__":
    network_retrieval = NetworkRetrieval("IUBL_centered_at_centroid1")


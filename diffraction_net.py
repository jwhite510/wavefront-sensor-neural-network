import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import shutil
import tables


class GetData():
    def __init__(self, batch_size):
        self.batch_counter = 0
        self.batch_index = 0
        self.batch_size = batch_size
        self.train_filename = "train_data.hdf5"
        self.test_filename = "test_data.hdf5"
        self.hdf5_file_train = tables.open_file(self.train_filename, mode="r")
        self.hdf5_file_validation = tables.open_file(self.test_filename, mode="r")
        self.samples = self.hdf5_file_train.root.object_amplitude.shape[0]
        # shape of the sample
        self.N = self.hdf5_file_train.root.N[0,0]
        print("initializing GetData")
        print("self.N =>", self.N)
        print("self.samples =>", self.samples)

    def next_batch(self):
        # retrieve the next batch of data from the data source
        samples = {}
        samples["object_amplitude_samples"] = self.hdf5_file_train.root.object_amplitude[self.batch_index:self.batch_index + self.batch_size, :]
        samples["object_phase_samples"] = self.hdf5_file_train.root.object_phase[self.batch_index:self.batch_index + self.batch_size, :]
        samples["diffraction_samples"] = self.hdf5_file_train.root.diffraction[self.batch_index:self.batch_index + self.batch_size, :]

        self.batch_index += self.batch_size

        return  samples

    def evaluate_on_train_data(self, n_samples):
        samples = {}
        samples["object_amplitude_samples"] = self.hdf5_file_train.root.object_amplitude[:n_samples, :]
        samples["object_phase_samples"] = self.hdf5_file_train.root.object_phase[:n_samples, :]
        samples["diffraction_samples"] = self.hdf5_file_train.root.diffraction[:n_samples, :]

        return samples

    def evaluate_on_validation_data(self, n_samples):
        samples = {}
        samples["object_amplitude_samples"] = self.hdf5_file_validation.root.object_amplitude[:n_samples, :]
        samples["object_phase_samples"] = self.hdf5_file_validation.root.object_phase[:n_samples, :]
        samples["diffraction_samples"] = self.hdf5_file_validation.root.diffraction[:n_samples, :]

        return samples

    def __del__(self):
        self.hdf5_file_train.close()

class DiffractionNet():
    def __init__(self, name):
        self.name = name
        print("initializing network")
        print(name)

        # initialize get data object
        self.get_data = GetData(batch_size=10)

        # input image
        self.x = tf.placeholder(tf.float32, shape=[None, self.get_data.N , self.get_data.N, 1])
        # label
        self.y = tf.placeholder(tf.float32, shape=[None, self.get_data.N, self.get_data.N, 1])

        self.nodes = {}
        self.out = None
        self.out_logits = None
        self.setup_network()

        # learning rate
        self.s_LR = tf.placeholder(tf.float32, shape=[])
        # define loss function


        # # testing output
        # self.out_logits = tf.constant(np.array([[0.0, 1.0, 0.0, 0.0]]))
        # self.out = tf.constant(np.array([[0.01, 1.0, 0.01, 0.01]]))
        # self.y = tf.constant(np.array([[0.0, 0.0, 1.0, 0.0]]))


        # # mean squared error
        # self.loss = tf.losses.mean_squared_error(labels=self.y, predictions=self.out)


        # # original cost function i used (after mean_squared_error)
        # self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y, logits=self.out_logits))


        # # identical cost function #1
        # sm = tf.nn.softmax(self.out)
        # self.loss = -tf.reduce_sum(self.y * tf.log(sm))


        # # identical cost function #2
        # self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y, logits=self.out)


        # # without softmax activation
        # self.loss = -tf.reduce_sum(self.y * tf.log(self.out))


        # still dont understand these
        # loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.out)
        # loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y, logits=self.out))

        # with tf.Session() as sess:
            # out = sess.run(self.loss)
            # print("out =>", out)
        # exit()

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.s_LR)
        self.train = self.optimizer.minimize(self.loss)

        # save file
        if not os.path.isdir('./models'):
            os.makedirs('./models')
        shutil.copyfile('./'+__file__, './models/'+__file__.split(".")[0]+'_{}.py'.format(self.name))

        # setup logging
        self.tf_loggers = {}
        self.setup_logging()

        # initialize graph
        self.init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(self.init)
        self.writer = tf.summary.FileWriter("./tensorboard_graph/" + self.name)

        # number of epochs to run
        self.epochs = 80
        self.i = None
        self.epoch = None
        self.dots = None

    def setup_network(self):
        # convolutional layer down sampling

        # self.nodes["conv1"] = convolutional_layer(self.x, shape=[3,3,1,32], activate='relu', stride=[1,1])
        self.nodes["conv1"] = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='SAME', activation='relu')(self.x)

        # self.nodes["conv2"] = convolutional_layer(self.nodes["conv1"], shape=[3,3,32,32], activate='relu', stride=[1,1])
        self.nodes["conv2"] = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='SAME', activation='relu')(self.nodes["conv1"])

        # max pooling
        # self.nodes["pool3"] = max_pooling_layer(self.nodes["conv2"], pool_size_val=[2,2], stride_val=[2,2], pad=True)
        self.nodes["pool3"] = tf.keras.layers.MaxPooling2D(pool_size=(2,2), padding='SAME')(self.nodes["conv2"])

        # convolutional layer
        # self.nodes["conv4"] = convolutional_layer(self.nodes["pool3"], shape=[3,3,32,64], activate='relu', stride=[1,1])
        self.nodes["conv4"] = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='SAME', activation='relu')(self.nodes["pool3"])

        # self.nodes["conv5"] = convolutional_layer(self.nodes["conv4"], shape=[3,3,64,64], activate='relu', stride=[1,1])
        self.nodes["conv5"] = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='SAME', activation='relu')(self.nodes["conv4"])

        # max pooling
        # self.nodes["pool6"] = max_pooling_layer(self.nodes["conv5"], pool_size_val=[2,2], stride_val=[2,2], pad=True)
        self.nodes["pool6"] = tf.keras.layers.MaxPooling2D(pool_size=(2,2), padding='SAME')(self.nodes["conv5"])

        # convolutional layer
        # self.nodes["conv7"] = convolutional_layer(self.nodes["pool6"], shape=[3,3,64,128], activate='relu', stride=[1,1])
        self.nodes["conv7"] = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='SAME', activation='relu')(self.nodes["pool6"])

        # self.nodes["conv8"] = convolutional_layer(self.nodes["conv7"], shape=[3,3,128,128], activate='relu', stride=[1,1])
        self.nodes["conv8"] = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='SAME', activation='relu')(self.nodes["conv7"])

        # max pooling
        # self.nodes["pool9"] = max_pooling_layer(self.nodes["conv8"], pool_size_val=[2,2], stride_val=[2,2], pad=True)
        self.nodes["pool9"] = tf.keras.layers.MaxPooling2D(pool_size=(2,2), padding='SAME')(self.nodes["conv8"])

        # convolutional layer
        # self.nodes["conv10"] = convolutional_layer(self.nodes["pool9"], shape=[3,3,128,128], activate='relu', stride=[1,1])
        self.nodes["conv10"] = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='SAME', activation='relu')(self.nodes["pool9"])

        # self.nodes["conv11"] = convolutional_layer(self.nodes["conv10"], shape=[3,3,128,128], activate='relu', stride=[1,1])
        self.nodes["conv11"] = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='SAME', activation='relu')(self.nodes["conv10"])

        # up sampling
        # self.nodes["ups12"] = upsample_2d(self.nodes["conv11"], 2)
        self.nodes["ups12"] = tf.keras.layers.UpSampling2D(size=2)(self.nodes["conv11"])

        # convolutional layer
        # self.nodes["conv13"] = convolutional_layer(self.nodes["ups12"], shape=[3,3,128,64], activate='relu', stride=[1,1])
        self.nodes["conv13"] = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='SAME', activation='relu')(self.nodes["ups12"])

        # self.nodes["conv14"] = convolutional_layer(self.nodes["conv13"], shape=[3,3,64,64], activate='relu', stride=[1,1])
        self.nodes["conv14"] = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='SAME', activation='relu')(self.nodes["conv13"])

        # up sampling
        # self.nodes["ups15"] = upsample_2d(self.nodes["conv14"], 2)
        self.nodes["ups15"] = tf.keras.layers.UpSampling2D(size=2)(self.nodes["conv14"])

        # convolutional layer
        # self.nodes["conv16"] = convolutional_layer(self.nodes["ups15"], shape=[3,3,64,32], activate='relu', stride=[1,1])
        self.nodes["conv16"] = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='SAME', activation='relu')(self.nodes["ups15"])

        # self.nodes["conv17"] = convolutional_layer(self.nodes["conv16"], shape=[3,3,32,32], activate='relu', stride=[1,1])
        self.nodes["conv17"] = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='SAME', activation='relu')(self.nodes["conv16"])

        # up sampling
        # self.nodes["ups18"] = upsample_2d(self.nodes["conv17"], 2)
        self.nodes["ups18"] = tf.keras.layers.UpSampling2D(size=2)(self.nodes["conv17"])

        # self.nodes["conv19"] = convolutional_layer(self.nodes["ups18"], shape=[3,3,32,1], activate='sigmoid', stride=[1,1])
        self.nodes["conv19"] = tf.keras.layers.Conv2D(filters=1, kernel_size=3, padding='SAME')(self.nodes["ups18"])

        # self.out = self.nodes["conv19"]
        self.out_logits = self.nodes["conv19"]

        self.out = tf.nn.sigmoid(self.out_logits)

    def setup_logging(self):
        self.tf_loggers["loss_training"] = tf.summary.scalar("loss", self.loss)
        self.tf_loggers["loss_validation"] = tf.summary.scalar("loss", self.loss)

    def supervised_learn(self):
        for self.i in range(self.epochs):
            self.epoch = self.i + 1
            print("Epoch : {}".format(self.epoch))
            self.dots = 0
            while self.get_data.batch_index < self.get_data.samples:
                self.show_loading_bar()

                # retrieve data
                data = self.get_data.next_batch()

                # run training iteration
                object_amplitude_samples = data["object_amplitude_samples"].reshape(-1,self.get_data.N, self.get_data.N, 1)
                object_phase_samples = data["object_phase_samples"].reshape(-1,self.get_data.N, self.get_data.N, 1)
                diffraction_samples = data["diffraction_samples"].reshape(-1,self.get_data.N, self.get_data.N, 1)

                self.sess.run(self.train, feed_dict={self.x:diffraction_samples,
                                                    self.y:object_amplitude_samples,
                                                    self.s_LR:0.0001})
            self.add_tensorboard_values()
            if self.i % 5 == 0:
                # plot the output
                output = self.sess.run(self.out, feed_dict={self.x:diffraction_samples})
                self.epoch

                # create directory if it doesnt exist
                if not os.path.isdir("nn_pictures"):
                    os.mkdir("nn_pictures")
                if not os.path.isdir("nn_pictures/"+self.name+"_pictures"):
                    os.mkdir("nn_pictures/"+self.name+"_pictures")
                if not os.path.isdir("nn_pictures/"+self.name+"_pictures/"+str(self.epoch)):
                    os.mkdir("nn_pictures/"+self.name+"_pictures/"+str(self.epoch))

                for index in range(0,5):
                    axes_obj = PlotAxes("sample "+str(index))
                    axes_obj.diffraction_input.pcolormesh(diffraction_samples[index,:,:,0])
                    axes_obj.object_actual.pcolormesh(object_amplitude_samples[index,:,:,0])
                    axes_obj.object_output.pcolormesh(output[index,:,:,0])
                    # axes_obj.diffraction_recons.pcolormesh()
                    axes_obj.save("nn_pictures/"+self.name+"_pictures/"+str(self.epoch)+"/sample_"+str(index))
                    del axes_obj



            self.get_data.batch_index = 0

    def add_tensorboard_values(self):
        # # # # # # # # # # # # # # #
        # loss on the training data #
        # # # # # # # # # # # # # # #
        data = self.get_data.evaluate_on_train_data(n_samples=50)
        object_amplitude_samples = data["object_amplitude_samples"].reshape(-1,self.get_data.N, self.get_data.N, 1)
        object_phase_samples = data["object_phase_samples"].reshape(-1,self.get_data.N, self.get_data.N, 1)
        diffraction_samples = data["diffraction_samples"].reshape(-1,self.get_data.N, self.get_data.N, 1)
        loss_value = self.sess.run(self.loss, feed_dict={self.x:diffraction_samples, self.y:object_amplitude_samples})
        print("training loss_value =>", loss_value)

        # write to log
        summ = self.sess.run(self.tf_loggers["loss_training"], feed_dict={self.x:diffraction_samples, self.y:object_amplitude_samples})
        self.writer.add_summary(summ, global_step=self.epoch)
        self.writer.flush()

        # # # # # # # # # # # # # # # #
        # loss on the validation data #
        # # # # # # # # # # # # # # # #
        data = self.get_data.evaluate_on_validation_data(n_samples=50)
        object_amplitude_samples = data["object_amplitude_samples"].reshape(-1,self.get_data.N, self.get_data.N, 1)
        object_phase_samples = data["object_phase_samples"].reshape(-1,self.get_data.N, self.get_data.N, 1)
        diffraction_samples = data["diffraction_samples"].reshape(-1,self.get_data.N, self.get_data.N, 1)
        loss_value = self.sess.run(self.loss, feed_dict={self.x:diffraction_samples, self.y:object_amplitude_samples})
        print("validation loss_value =>", loss_value)

        # write to log
        summ = self.sess.run(self.tf_loggers["loss_validation"], feed_dict={self.x:diffraction_samples, self.y:object_amplitude_samples})
        self.writer.add_summary(summ, global_step=self.epoch)
        self.writer.flush()

    def __del__(self):
        del self.get_data

    def show_loading_bar(self):
        # display loading bar
        percent = 50 * self.get_data.batch_index / self.get_data.samples
        if percent - self.dots > 1:
            print(".", end="", flush=True)
            self.dots += 1

def max_pooling_layer(input_x, pool_size_val,  stride_val, pad=False):
    if pad:
        return tf.layers.max_pooling2d(input_x, pool_size=[pool_size_val[0], pool_size_val[1]], strides=[stride_val[0], stride_val[1]], padding="SAME")
    else:
        return tf.layers.max_pooling2d(input_x, pool_size=[pool_size_val[0], pool_size_val[1]], strides=[stride_val[0], stride_val[1]], padding="VALID")

def init_weights(shape):
    init_random_dist = tf.truncated_normal(shape, stddev=0.1, dtype=tf.float32)
    return tf.Variable(init_random_dist)

def init_bias(shape):
    init_bias_vals = tf.constant(0.1, shape=shape, dtype=tf.float32)
    return tf.Variable(init_bias_vals)

def convolutional_layer(input_x, shape, activate, stride):
    W = init_weights(shape)
    b = init_bias([shape[3]])

    if activate == 'relu':
        return tf.nn.relu(conv2d(input_x, W, stride) + b)

    if activate == 'leaky':
        return tf.nn.leaky_relu(conv2d(input_x, W, stride) + b)

    if activate == 'sigmoid':
        return tf.nn.sigmoid(conv2d(input_x, W, stride) + b)

    elif activate == 'none':
        return conv2d(input_x, W, stride) + b

def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride[0], stride[1], 1], padding='SAME')

def upsample_2d(x, S):
    height = int(x.shape[1])
    width = int(x.shape[2])

    return tf.image.resize_nearest_neighbor(x, (S*height, S*width))



class PlotAxes():
    def __init__(self,fig_title):
        """
        fig_title:(string)
        title for the figure
        """
        # create plot
        self.fig = plt.figure(figsize=(10,10))
        self.gs = self.fig.add_gridspec(2,2)

        self.fig.text(0.5, 0.95,fig_title, fontsize=30, ha='center')

        self.axes = {}
        self.diffraction_input = self.fig.add_subplot(self.gs[0:1,0:1])
        self.diffraction_input.set_title("diffraction_input")
        self.diffraction_input.set_xticks([])
        self.diffraction_input.set_yticks([])

        self.diffraction_recons = self.fig.add_subplot(self.gs[0:1,1:2])
        self.diffraction_recons.set_title("diffraction_recons")
        self.diffraction_recons.set_xticks([])
        self.diffraction_recons.set_yticks([])

        self.object_actual = self.fig.add_subplot(self.gs[1:2,1:2])
        self.object_actual.set_title("object_actual")
        self.object_actual.set_xticks([])
        self.object_actual.set_yticks([])

        self.object_output = self.fig.add_subplot(self.gs[1:2,0:1])
        self.object_output.set_title("object_output")
        self.object_output.set_xticks([])
        self.object_output.set_yticks([])

    def save(self, filename):
        self.fig.savefig(filename)


if __name__ == "__main__":
    # getdata = GetData(batch_size=10)
    # getdata.next_batch()
    # del getdata

    diffraction_net = DiffractionNet(name="test8")
    diffraction_net.supervised_learn()
    del diffraction_net
    # pass





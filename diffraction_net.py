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
        self.hdf5_file = tables.open_file(self.train_filename, mode="r")
        self.samples = self.hdf5_file.root.object_amplitude.shape[0]
        # shape of the sample
        self.N = self.hdf5_file.root.N[0,0]
        print("initializing GetData")
        print("self.N =>", self.N)
        print("self.samples =>", self.samples)

    def next_batch(self):
        # retrieve the next batch of data from the data source
        samples = {}
        samples["object_amplitude_samples"] = self.hdf5_file.root.object_amplitude[0,:]
        samples["object_phase_samples"] = self.hdf5_file.root.object_phase[0, :]
        samples["diffraction_samples"] = self.hdf5_file.root.diffraction[0, :]

        self.batch_index += self.batch_size

        return  samples

    def evaluate_on_train_data(self, n_samples):
        samples = {}
        samples["object_amplitude_samples"] = self.hdf5_file.root.object_amplitude[:n_samples, :]
        samples["object_phase_samples"] = self.hdf5_file.root.object_phase[:n_samples, :]
        samples["diffraction_samples"] = self.hdf5_file.root.diffraction[:n_samples, :]

        return samples

    def __del__(self):
        self.hdf5_file.close()

class DiffractionNet():
    def __init__(self, name):
        self.name = name

        # initialize get data object
        self.get_data = GetData(batch_size=10)

        # input image
        self.x = tf.placeholder(tf.float32, shape=[None, self.get_data.N , self.get_data.N, 1])
        # label
        self.y = tf.placeholder(tf.float32, shape=[None, self.get_data.N, self.get_data.N, 1])

        self.nodes = {}
        self.out = None
        self.setup_network()

        # learning rate
        self.s_LR = tf.placeholder(tf.float32, shape=[])
        # define loss function
        self.loss = tf.losses.mean_squared_error(labels=self.y, predictions=self.out)
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
        self.nodes["conv1"] = convolutional_layer(self.x, shape=[3,3,1,32], activate='relu', stride=[1,1])
        layer1 = tf.contrib.layers.flatten(self.nodes["conv1"])
        layer2 = tf.layers.dense(inputs=layer1, units=128)
        layer3 = tf.layers.dense(inputs=layer2, units=40*40)
        layer3_sh = tf.reshape(layer3, [-1, 40, 40, 1])
        self.out = layer3_sh

        # self.nodes["conv2"] = convolutional_layer(self.nodes["conv1"], shape=[3,3,32,32], activate='relu', stride=[1,1])
        # # max pooling
        # self.nodes["pool3"] = max_pooling_layer(self.nodes["conv2"], pool_size_val=[2,2], stride_val=[2,2], pad=True)
        # # convolutional layer
        # self.nodes["conv4"] = convolutional_layer(self.nodes["pool3"], shape=[3,3,32,64], activate='relu', stride=[1,1])
        # self.nodes["conv5"] = convolutional_layer(self.nodes["conv4"], shape=[3,3,64,64], activate='relu', stride=[1,1])
        # # max pooling
        # self.nodes["pool6"] = max_pooling_layer(self.nodes["conv5"], pool_size_val=[2,2], stride_val=[2,2], pad=True)
        # # convolutional layer
        # self.nodes["conv7"] = convolutional_layer(self.nodes["pool6"], shape=[3,3,64,128], activate='relu', stride=[1,1])
        # self.nodes["conv8"] = convolutional_layer(self.nodes["conv7"], shape=[3,3,128,128], activate='relu', stride=[1,1])
        # # max pooling
        # self.nodes["pool9"] = max_pooling_layer(self.nodes["conv8"], pool_size_val=[2,2], stride_val=[2,2], pad=True)
        # # convolutional layer
        # self.nodes["conv10"] = convolutional_layer(self.nodes["pool9"], shape=[3,3,128,128], activate='relu', stride=[1,1])
        # self.nodes["conv11"] = convolutional_layer(self.nodes["conv10"], shape=[3,3,128,128], activate='relu', stride=[1,1])
        # # up sampling
        # self.nodes["ups12"] = upsample_2d(self.nodes["conv11"], 2)
        # # convolutional layer
        # self.nodes["conv13"] = convolutional_layer(self.nodes["ups12"], shape=[3,3,128,64], activate='relu', stride=[1,1])
        # self.nodes["conv14"] = convolutional_layer(self.nodes["conv13"], shape=[3,3,64,64], activate='relu', stride=[1,1])
        # # up sampling
        # self.nodes["ups15"] = upsample_2d(self.nodes["conv14"], 2)
        # # convolutional layer
        # self.nodes["conv16"] = convolutional_layer(self.nodes["ups15"], shape=[3,3,64,32], activate='relu', stride=[1,1])
        # self.nodes["conv17"] = convolutional_layer(self.nodes["conv16"], shape=[3,3,32,32], activate='relu', stride=[1,1])
        # # up sampling
        # self.nodes["ups18"] = upsample_2d(self.nodes["conv17"], 2)
        # self.nodes["conv19"] = convolutional_layer(self.nodes["ups18"], shape=[3,3,32,1], activate='sigmoid', stride=[1,1])
        # self.out = self.nodes["conv19"]

    def setup_logging(self):
        self.tf_loggers["loss"] = tf.summary.scalar("loss", self.loss)

    def supervised_learn(self):
        print("train network")
        plt.ion()
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
                # print("self.x =>", self.x)
                # print("self.y =>", self.y)

                self.sess.run(self.train, feed_dict={self.x:diffraction_samples,
                                                    self.y:object_amplitude_samples,
                                                    self.s_LR:0.001})
                self.add_tensorboard_values()

            if self.i % 2 == 0:
                # look at the output
                output = self.sess.run(self.out, feed_dict={self.x:diffraction_samples})

                index = 0
                plt.figure(1)
                plt.gca().cla()
                plt.pcolormesh(object_amplitude_samples[index,:,:,0])
                plt.figure(2)
                plt.gca().cla()
                plt.pcolormesh(output[index,:,:,0])
                plt.pause(0.1)


            self.get_data.batch_index = 0

    def add_tensorboard_values(self):
        data = self.get_data.evaluate_on_train_data(n_samples=50)
        object_amplitude_samples = data["object_amplitude_samples"].reshape(-1,self.get_data.N, self.get_data.N, 1)
        object_phase_samples = data["object_phase_samples"].reshape(-1,self.get_data.N, self.get_data.N, 1)
        diffraction_samples = data["diffraction_samples"].reshape(-1,self.get_data.N, self.get_data.N, 1)
        loss_value = self.sess.run(self.loss, feed_dict={self.x:diffraction_samples, self.y:object_amplitude_samples})
        print("loss_value =>", loss_value)

        # write to log
        summ = self.sess.run(self.tf_loggers["loss"], feed_dict={self.x:diffraction_samples, self.y:object_amplitude_samples})
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

if __name__ == "__main__":
    # getdata = GetData(batch_size=10)
    # getdata.next_batch()
    # del getdata

    diffraction_net = DiffractionNet(name="test2")
    diffraction_net.supervised_learn()
    del diffraction_net
    # pass





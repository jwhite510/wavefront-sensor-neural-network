import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import shutil


class DiffractionNet():
    def __init__(self, name, N):
        self.name = name
        # input image
        self.x = tf.placeholder(tf.float32, shape=[None, N, N, 1])
        # label
        self.y = tf.placeholder(tf.float32, shape=[None, N, N, 1])

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

        shutil.copyfile('./'+__file__, './models/'+__file__+'_{}.py'.format(self.name))

        # setup logging
        self.tf_loggers = {}
        self.setup_logging()

        # initialize graph
        self.init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(self.init)
        self.writer = tf.summary.FileWriter("./tensorboard_graph/" + self.name)

    def setup_network(self):

        # convolutional layer down sampling
        self.nodes["conv1"] = convolutional_layer(self.x, shape=[3,3,1,32], activate='relu', stride=[1,1])
        self.nodes["conv2"] = convolutional_layer(self.nodes["conv1"], shape=[3,3,32,32], activate='relu', stride=[1,1])
        # max pooling
        self.nodes["pool3"] = max_pooling_layer(self.nodes["conv2"], pool_size_val=[2,2], stride_val=[2,2], pad=True)
        # convolutional layer
        self.nodes["conv4"] = convolutional_layer(self.nodes["pool3"], shape=[3,3,32,64], activate='relu', stride=[1,1])
        self.nodes["conv5"] = convolutional_layer(self.nodes["conv4"], shape=[3,3,64,64], activate='relu', stride=[1,1])
        # max pooling
        self.nodes["pool6"] = max_pooling_layer(self.nodes["conv5"], pool_size_val=[2,2], stride_val=[2,2], pad=True)
        # convolutional layer
        self.nodes["conv7"] = convolutional_layer(self.nodes["pool6"], shape=[3,3,64,128], activate='relu', stride=[1,1])
        self.nodes["conv8"] = convolutional_layer(self.nodes["conv7"], shape=[3,3,128,128], activate='relu', stride=[1,1])
        # max pooling
        self.nodes["pool9"] = max_pooling_layer(self.nodes["conv8"], pool_size_val=[2,2], stride_val=[2,2], pad=True)
        # convolutional layer
        self.nodes["conv10"] = convolutional_layer(self.nodes["pool9"], shape=[3,3,128,128], activate='relu', stride=[1,1])
        self.nodes["conv11"] = convolutional_layer(self.nodes["conv10"], shape=[3,3,128,128], activate='relu', stride=[1,1])
        # up sampling
        self.nodes["ups12"] = upsample_2d(self.nodes["conv11"], 2)
        # convolutional layer
        self.nodes["conv13"] = convolutional_layer(self.nodes["ups12"], shape=[3,3,128,64], activate='relu', stride=[1,1])
        self.nodes["conv14"] = convolutional_layer(self.nodes["conv13"], shape=[3,3,64,64], activate='relu', stride=[1,1])
        # up sampling
        self.nodes["ups15"] = upsample_2d(self.nodes["conv14"], 2)
        # convolutional layer
        self.nodes["conv16"] = convolutional_layer(self.nodes["ups15"], shape=[3,3,64,32], activate='relu', stride=[1,1])
        self.nodes["conv17"] = convolutional_layer(self.nodes["conv16"], shape=[3,3,32,32], activate='relu', stride=[1,1])
        # up sampling
        self.nodes["ups18"] = upsample_2d(self.nodes["conv17"], 2)
        self.nodes["conv19"] = convolutional_layer(self.nodes["ups18"], shape=[3,3,32,1], activate='sigmoid', stride=[1,1])

        self.out = self.nodes["conv19"]

    def setup_logging(self):
        self.tf_loggers["loss"] = tf.summary.scalar("loss", self.loss)

    def train(self):
        pass


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

    diffraction_net = DiffractionNet(name="test1", N=32)
    pass





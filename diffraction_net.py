import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class DiffractionNet():
    def __init__(self, N):
        self.x = tf.placeholder(tf.float32, shape=[None, N, N, 1])

        # convolutional layer down sampling
        conv1 = convolutional_layer(self.x, shape=[3,3,1,32], activate='relu', stride=[1,1])
        conv2 = convolutional_layer(conv1, shape=[3,3,32,32], activate='relu', stride=[1,1])
        # max pooling
        pool3 = max_pooling_layer(conv2, pool_size_val=[2,2], stride_val=[2,2], pad=True)
        exit()
        import ipdb; ipdb.set_trace() # BREAKPOINT
        print("BREAKPOINT")


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

    elif activate == 'none':
        return conv2d(input_x, W, stride) + b


def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride[0], stride[1], 1], padding='SAME')

def UpSample2D(x, S):

    return tf.image.resize_nearest_neighbor(x, (S*height, S*width))





if __name__ == "__main__":

    diffraction_net = DiffractionNet(N=32)
    pass





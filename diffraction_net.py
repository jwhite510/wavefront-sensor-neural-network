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
        # convolutional layer
        conv4 = convolutional_layer(pool3, shape=[3,3,32,64], activate='relu', stride=[1,1])
        conv5 = convolutional_layer(conv4, shape=[3,3,64,64], activate='relu', stride=[1,1])
        # max pooling
        pool6 = max_pooling_layer(conv5, pool_size_val=[2,2], stride_val=[2,2], pad=True)
        # convolutional layer
        conv7 = convolutional_layer(pool6, shape=[3,3,64,128], activate='relu', stride=[1,1])
        conv8 = convolutional_layer(conv7, shape=[3,3,128,128], activate='relu', stride=[1,1])
        # max pooling
        pool9 = max_pooling_layer(conv8, pool_size_val=[2,2], stride_val=[2,2], pad=True)
        # convolutional layer
        conv10 = convolutional_layer(pool9, shape=[3,3,128,128], activate='relu', stride=[1,1])
        conv11 = convolutional_layer(conv10, shape=[3,3,128,128], activate='relu', stride=[1,1])
        # up sampling
        ups12 = upsample_2d(conv11, 2)
        # convolutional layer
        conv13 = convolutional_layer(ups12, shape=[3,3,128,64], activate='relu', stride=[1,1])
        conv14 = convolutional_layer(conv13, shape=[3,3,64,64], activate='relu', stride=[1,1])
        # up sampling
        ups15 = upsample_2d(conv14, 2)
        # convolutional layer
        conv16 = convolutional_layer(ups15, shape=[3,3,64,32], activate='relu', stride=[1,1])
        conv17 = convolutional_layer(conv16, shape=[3,3,32,32], activate='relu', stride=[1,1])
        # up sampling
        ups18 = upsample_2d(conv17, 2)
        conv19 = convolutional_layer(ups18, shape=[3,3,32,1], activate='sigmoid', stride=[1,1])

        print("conv1 =>", conv1)
        print("conv2 =>", conv2)
        print("pool3 =>", pool3)
        print("conv4 =>", conv4)
        print("conv5 =>", conv5)
        print("pool6 =>", pool6)
        print("conv7 =>", conv7)
        print("conv8 =>", conv8)
        print("pool9 =>", pool9)
        print("conv10 =>", conv10)
        print("conv11 =>", conv11)
        print("ups12 =>", ups12)
        print("conv13 =>", conv13)
        print("conv14 =>", conv14)
        print("ups15 =>", ups15)
        print("conv16 =>", conv16)
        print("conv17 =>", conv17)
        print("ups18 =>", ups18)
        print("conv19 =>", conv19)

        exit()


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

    diffraction_net = DiffractionNet(N=32)
    pass





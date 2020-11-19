import tensorflow as tf

def convolutional_layer_nopadding(input_x, shape, activate, stride):
    W = init_weights(shape)
    b = init_bias([shape[3]])

    if activate == 'relu':
        return tf.nn.relu(conv2d_nopad(input_x, W, stride) + b)

    if activate == 'leaky':
        return tf.nn.leaky_relu(conv2d_nopad(input_x, W, stride) + b)

    elif activate == 'none':
        return conv2d_nopad(input_x, W, stride) + b

def max_pooling_layer(input_x, pool_size_val,  stride_val, pad=False):
    if pad:
        return tf.layers.max_pooling2d(input_x, pool_size=[pool_size_val[0], pool_size_val[1]], strides=[stride_val[0], stride_val[1]], padding="SAME")
    else:
        return tf.layers.max_pooling2d(input_x, pool_size=[pool_size_val[0], pool_size_val[1]], strides=[stride_val[0], stride_val[1]], padding="VALID")

def part1_purple(input):
    conv1 = convolutional_layer_nopadding(input, shape=[10, 10, 1, 20], activate='relu', stride=[1, 1])
    print("conv1", conv1)

    pool1 = max_pooling_layer(conv1, pool_size_val=[6, 6], stride_val=[3, 3])
    print("pool1", pool1)

    conv2 = convolutional_layer_nopadding(pool1, shape=[6, 6, 20, 40], activate='relu', stride=[1, 1])
    print("conv2", conv2)

    pool2 = max_pooling_layer(conv2, pool_size_val=[4, 4], stride_val=[2, 2])
    print("pool2 =>", pool2)

    return pool2

def part2_grey(input):
    # center
    conv31 = convolutional_layer_nopadding(input, shape=[6, 6, 40, 40], activate='relu', stride=[1, 1])
    conv322 = convolutional_layer_nopadding(conv31, shape=[6, 6, 40, 20], activate='relu', stride=[1, 1])

    #left
    pool3 = max_pooling_layer(input, pool_size_val=[7, 7], stride_val=[2, 2])
    conv321 = convolutional_layer_nopadding(pool3, shape=[1, 1, 40, 20], activate='relu', stride=[1, 1])

    #right
    conv323 = convolutional_layer_nopadding(input, shape=[11, 11, 40, 20], activate='relu', stride=[1, 1])

    conc1 = tf.concat([conv321, conv322, conv323], axis=3)

    return conc1

def part3_green(input):
    # left side
    conv4 = convolutional_layer_nopadding(input, shape=[3, 3, 60, 40], activate='relu', stride=[1, 1])

    # right side
    pool4 = max_pooling_layer(input, pool_size_val=[3, 3], stride_val=[1, 1])

    # concatinate
    conc2 = tf.concat([conv4, pool4], axis=3)

    return conc2

def avg_pooling_layer(input_x, pool_size_val,  stride_val, pad=False):
    if pad:
        return tf.layers.average_pooling2d(input_x, pool_size=[pool_size_val[0], pool_size_val[1]], strides=[stride_val[0], stride_val[1]], padding="SAME")
    else:
        return tf.layers.average_pooling2d(input_x, pool_size=[pool_size_val[0], pool_size_val[1]], strides=[stride_val[0], stride_val[1]], padding="VALID")

def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride[0], stride[1], 1], padding='SAME')

def conv2d_nopad(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride[0], stride[1], 1], padding="VALID")

def init_weights(shape):
    init_random_dist = tf.truncated_normal(shape, stddev=0.1, dtype=tf.float32)
    return tf.Variable(init_random_dist)

def init_bias(shape):
    init_bias_vals = tf.constant(0.1, shape=shape, dtype=tf.float32)
    return tf.Variable(init_bias_vals)

def normal_full_layer(input_layer, size):
    input_size = int(input_layer.get_shape()[1])
    W = init_weights([input_size, size])
    b = init_bias([size])
    return tf.matmul(input_layer, W) + b

def noise_resistant_phase_retrieval_net(input_image:tf.Tensor,zernike_coefs:int)->(tf.Tensor,tf.Tensor,tf.Tensor):

    with tf.variable_scope("phase"):
        pool2 = part1_purple(input_image)

        conc1 = part2_grey(pool2)

        conc2 = part3_green(conc1)

        pool51 = avg_pooling_layer(conc2, pool_size_val=[3, 3], stride_val=[1, 1])
        # print("pool51", pool51)

        pool52 = avg_pooling_layer(pool2, pool_size_val=[5, 5], stride_val=[5, 5], pad=True)
        # print("pool52", pool52)

        pool53 = avg_pooling_layer(conc1, pool_size_val=[3, 3], stride_val=[2, 2])
        # print("pool53", pool53)

        pool51_flat = tf.contrib.layers.flatten(pool51)
        pool52_flat = tf.contrib.layers.flatten(pool52)
        pool53_flat = tf.contrib.layers.flatten(pool53)

        conc3 = tf.concat([pool51_flat, pool52_flat, pool53_flat], axis=1)

        fc5 = tf.layers.dense(inputs=conc3, units=256)

        # dropout
        hold_prob = tf.placeholder_with_default(1.0, shape=())
        dropout_layer = tf.nn.dropout(fc5, keep_prob=hold_prob)

        # output layer
        predicted_zernike = normal_full_layer(dropout_layer, zernike_coefs)
        predicted_scale = normal_full_layer(dropout_layer, 1)

        return predicted_zernike, predicted_scale, hold_prob

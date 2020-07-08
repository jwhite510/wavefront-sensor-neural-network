import tensorflow as tf

def init_weights(shape):
    init_random_dist = tf.truncated_normal(shape, stddev=0.1, dtype=tf.float32)
    return tf.Variable(init_random_dist)

def init_bias(shape):
    init_bias_vals = tf.constant(0.1, shape=shape, dtype=tf.float32)
    return tf.Variable(init_bias_vals)

def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride[0], stride[1], 1], padding='SAME')

def convolutional_layer(input_x, shape, activate, stride):
    W = init_weights(shape)
    b = init_bias([shape[3]])

    if activate == 'relu':
        return tf.nn.relu(conv2d(input_x, W, stride) + b)

    if activate == 'leaky':
        return tf.nn.leaky_relu(conv2d(input_x, W, stride) + b)

    elif activate == 'none':
        return conv2d(input_x, W, stride) + b

def multires_layer(input, input_channels, filter_sizes, stride=1):
    # list of layers
    filters = []
    for filter_size in filter_sizes:
        # create filter
        filters.append(convolutional_layer(input, shape=[filter_size, filter_size,
                        input_channels, input_channels], activate='leaky', stride=[stride, stride]))
        #TODO try with relu also

    concat_layer = tf.concat(filters, axis=3)
    return concat_layer

def reverse_multires_layer(input, input_channels, filter_sizes, stride, n_of_each_filter):
    # list of layers
    filters = []
    for filter_size in filter_sizes:
        # create filter

        f = tf.keras.layers.Conv2DTranspose(filters=n_of_each_filter, kernel_size=filter_size, padding='SAME', strides=stride, activation='relu')(input)
        filters.append(f)

    concat_layer = tf.concat(filters, axis=3)
    return concat_layer

import tensorflow as tf

def lrelu(x, alpha=0.3):
    return tf.maximum(x, tf.multiply(x, alpha))

def encoder(X_in, n_coefs):
    keep_prob = tf.placeholder_with_default(1.0, shape=())
    activation = lrelu
    # X = tf.reshape(X_in, shape=[-1, 28, 28, 1])
    x = tf.layers.conv2d(X_in, filters=64, kernel_size=8, strides=2, padding='same', activation=activation)
    x = tf.nn.dropout(x, keep_prob)
    x = tf.layers.conv2d(x, filters=64, kernel_size=8, strides=2, padding='same', activation=activation)
    x = tf.nn.dropout(x, keep_prob)
    x = tf.layers.conv2d(x, filters=64, kernel_size=8, strides=2, padding='same', activation=activation)
    x = tf.nn.dropout(x, keep_prob)
    x = tf.layers.conv2d(x, filters=32, kernel_size=8, strides=2, padding='same', activation=activation)
    x = tf.nn.dropout(x, keep_prob)
    x = tf.layers.conv2d(x, filters=8, kernel_size=8, strides=1, padding='same', activation=activation)
    x = tf.nn.dropout(x, keep_prob)
    x = tf.contrib.layers.flatten(x)
    _mean = tf.layers.dense(x, units=n_coefs+1)
    _gamma = tf.layers.dense(x, units=n_coefs+1) # additional 1 for scale number
    _sigma = tf.exp(0.5*_gamma)
    noise = tf.random_normal(tf.shape(_sigma))
    z = _mean + _sigma*noise
    _scale = tf.reshape(z[:,0],[-1,1])
    _coefs = z[:,1:]
    latent_loss = -0.5 * tf.reduce_sum(1.0 + 2.0 * _gamma - tf.square(_mean) - tf.exp(2.0 * _gamma))
    return _coefs, _scale, keep_prob, latent_loss

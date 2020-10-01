import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import diffraction_functions
import datagen

def make_dif_pattern(datagenerator:datagen.DataGenerator,coefs:tf.Tensor,scale:tf.Tensor)->tf.Tensor:
    beforewf=datagenerator.buildgraph(coefs,scale)
    afterwf=datagenerator.propagate_through_wfs(beforewf)
    diffraction_pattern = tf.abs(diffraction_functions.tf_fft2(afterwf, dimmensions=[1,2]))**2
    diffraction_pattern = diffraction_pattern / tf.reduce_max(diffraction_pattern, keepdims=True, axis=[1,2])
    return diffraction_pattern


if __name__ == "__main__":
    N=128
    datagenerator = datagen.DataGenerator(1024,N)
    coefs_actual = tf.placeholder(tf.float32, shape=[1, len(datagenerator.zernike_cvector)])
    scale_actual = tf.placeholder(tf.float32, shape=[1,1])
    diffraction_pattern_actual=make_dif_pattern(datagenerator,coefs_actual,scale_actual)

    # generate another diffraction pattern with variable
    coefs_guess = tf.Variable(tf.truncated_normal((1,14),stddev=0.1,dtype=tf.float32))
    coefs_guess = tf.sigmoid(coefs_guess)
    coefs_guess*=12
    coefs_guess-=6
    scale_guess = tf.Variable(tf.truncated_normal((1,1),stddev=0.1,dtype=tf.float32))
    scale_guess = tf.sigmoid(scale_guess)
    scale_guess+=0.5 # max 1.5, min 0.5
    diffraction_pattern_guess=make_dif_pattern(datagenerator,coefs_guess,scale_guess)
    # cost function
    # diffraction patterns should be similar
    diffraction_p_error=1000000*tf.losses.mean_squared_error(labels=diffraction_pattern_actual,predictions=diffraction_pattern_guess)
    coefs_error = tf.losses.mean_squared_error(labels=coefs_actual,predictions=coefs_guess)
    scale_error = tf.losses.mean_squared_error(labels=scale_actual,predictions=scale_guess)
    cost_function =  diffraction_p_error + -1*(coefs_error+scale_error)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    train=optimizer.minimize(cost_function)

    # logging
    tf_loggers={}
    tf_loggers["diffraction_p_error"] = tf.summary.scalar("diffraction_p_error", diffraction_p_error)
    tf_loggers["coefs_error"] = tf.summary.scalar("coefs_error", coefs_error)
    tf_loggers["scale_error"] = tf.summary.scalar("scale_error", scale_error)
    tf_loggers["cost_function"] = tf.summary.scalar("cost_function", cost_function)

    writer = tf.summary.FileWriter("./tensorboard_graph/" + "run10")

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        f = {
            coefs_actual:np.array([[0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]]),
            scale_actual:np.array([[1.0]]),
                }
        for i in range(1000):

            # add logs
            summ = sess.run(tf_loggers["diffraction_p_error"], feed_dict=f)
            writer.add_summary(summ, global_step=i)
            summ = sess.run(tf_loggers["coefs_error"], feed_dict=f)
            writer.add_summary(summ, global_step=i)
            summ = sess.run(tf_loggers["scale_error"], feed_dict=f)
            writer.add_summary(summ, global_step=i)
            summ = sess.run(tf_loggers["cost_function"], feed_dict=f)
            writer.add_summary(summ, global_step=i)
            writer.flush()

            _coefs_guess=sess.run(coefs_guess,feed_dict=f)
            _scale_guess=sess.run(scale_guess,feed_dict=f)
            _diffraction_pattern_guess=sess.run(diffraction_pattern_guess,feed_dict=f)
            print("i =>", i)
            print("_coefs_guess =>", _coefs_guess)
            print("_scale_guess =>", _scale_guess)
            sess.run(train, feed_dict=f)





    print("coefs_guess =>", coefs_guess)
    print("s_guess =>", scale_guess)
    print("coefs_actual =>", coefs_actual)
    print("scale_actual =>", scale_actual)





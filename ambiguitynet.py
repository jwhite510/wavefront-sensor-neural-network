import tensorflow as tf
import imageio
import numpy as np
import matplotlib.pyplot as plt
import diffraction_functions
import datagen

def make_dif_pattern(datagenerator:datagen.DataGenerator,coefs:tf.Tensor,scale:tf.Tensor)->(tf.Tensor,dict):
    beforewf=datagenerator.buildgraph(coefs,scale)
    afterwf=datagenerator.propagate_through_wfs(beforewf)
    diffraction_pattern = tf.abs(diffraction_functions.tf_fft2(afterwf, dimmensions=[1,2]))**2
    diffraction_pattern = diffraction_pattern / tf.reduce_max(diffraction_pattern, keepdims=True, axis=[1,2])
    return diffraction_pattern,{'beforewf':beforewf,'afterwf':afterwf}

def draw_figure(
        sess:tf.Session,
        diffraction_pattern_actual:np.array,
        coefs:tf.Tensor,
        scale:tf.Tensor,
        diffraction_pattern:tf.Tensor,
        obj:tf.Tensor,
        f:dict,
        title:str
        )->plt.figure:

    _coefs=sess.run(coefs,feed_dict=f)
    _scale=sess.run(scale,feed_dict=f)
    _diffraction_pattern=sess.run(diffraction_pattern,feed_dict=f)
    _obj=sess.run(obj,feed_dict=f)

    # draw the guess object
    fig=diffraction_functions.plot_amplitude_phase_meas_retreival(
            {
                "measured_pattern":np.squeeze(diffraction_pattern_actual),
                "tf_reconstructed_diff":np.squeeze(_diffraction_pattern),
                "real_output":np.real(np.squeeze(_obj['beforewf'])),
                "imag_output":np.imag(np.squeeze(_obj['beforewf'])),
                "coefficients":_coefs,
                "scale":_scale,
                },
            title,
            ACTUAL=True,
            mask=True
            )
    return fig



if __name__ == "__main__":
    N=128
    datagenerator = datagen.DataGenerator(1024,N)
    coefs_actual = tf.placeholder(tf.float32, shape=[1, len(datagenerator.zernike_cvector)])
    scale_actual = tf.placeholder(tf.float32, shape=[1,1])
    diffraction_pattern_actual,actual_obj=make_dif_pattern(datagenerator,coefs_actual,scale_actual)

    # generate another diffraction pattern with variable
    # coefs_guess = tf.Variable(tf.truncated_normal((1,14),stddev=0.1,dtype=tf.float32))
    coefs_guess = tf.Variable(np.array([14*[1]],dtype=np.float32))
    coefs_guess = tf.sigmoid(coefs_guess)
    coefs_guess*=12
    coefs_guess-=6
    scale_guess = tf.Variable(tf.truncated_normal((1,1),stddev=0.1,dtype=tf.float32))
    scale_guess = tf.sigmoid(scale_guess)
    scale_guess+=0.5 # max 1.5, min 0.5
    diffraction_pattern_guess,guess_obj=make_dif_pattern(datagenerator,coefs_guess,scale_guess)
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
            coefs_actual:np.array([[0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]]),
            scale_actual:np.array([[1.0]]),
                }
        gif_frames=[]
        error_vals={
                "diffraction_p_error":[],
                "coefs_error":[],
                "scale_error":[],
                "cost_function":[],
                }
        i_max=250
        for i in range(i_max):
            print(i)

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
            error_vals['diffraction_p_error'].append(sess.run(diffraction_p_error,feed_dict=f))
            error_vals['coefs_error'].append(sess.run(coefs_error,feed_dict=f))
            error_vals['scale_error'].append(sess.run(scale_error,feed_dict=f))
            error_vals['cost_function'].append(sess.run(cost_function,feed_dict=f))

            # get actual diffraction pattern
            _diffraction_pattern_actual=sess.run(diffraction_pattern_actual,feed_dict=f)

            fig=draw_figure(sess,_diffraction_pattern_actual,coefs_guess,scale_guess,diffraction_pattern_guess,guess_obj,f,"GUESS "+str(i))
            fig.canvas.draw()
            image_guess=np.frombuffer(fig.canvas.tostring_rgb(),dtype='uint8')
            image_guess=image_guess.reshape(fig.canvas.get_width_height()[::-1]+(3,))
            plt.close(fig)

            fig=draw_figure(sess,_diffraction_pattern_actual,coefs_actual,scale_actual,diffraction_pattern_actual,actual_obj,f,"ACTUAL")
            fig.canvas.draw()
            image_actual=np.frombuffer(fig.canvas.tostring_rgb(),dtype='uint8')
            image_actual=image_actual.reshape(fig.canvas.get_width_height()[::-1]+(3,))
            plt.close(fig)


            # draw plot
            fig = plt.figure(figsize=(20,3))
            gs = fig.add_gridspec(1,1)
            ax=fig.add_subplot(gs[0,0])
            ax.plot(error_vals['diffraction_p_error'],label='diffraction_p_error')
            ax.plot(error_vals['coefs_error'],label='coefs_error')
            ax.plot(error_vals['scale_error'],label='scale_error')
            ax.plot(error_vals['cost_function'],label='cost_function')
            ax.set_xlim(0,i_max)
            ax.text(0.5,-0.1,"Cost Function:"+"%.4f"%sess.run(cost_function,feed_dict=f),ha='center',transform=ax.transAxes,backgroundcolor='red')
            ax.legend()
            fig.canvas.draw()
            im_plot=np.frombuffer(fig.canvas.tostring_rgb(),dtype='uint8')
            im_plot=im_plot.reshape(fig.canvas.get_width_height()[::-1]+(3,))
            plt.close(fig)

            # draw the actual object
            image_both=np.append(image_actual,image_guess,axis=1)
            image_both = np.append(im_plot,image_both,axis=0)
            gif_frames.append(image_both)

            # append the plots

            # train
            sess.run(train, feed_dict=f)

        imageio.mimsave('./'+'file_large.gif',gif_frames,fps=10)





    print("coefs_guess =>", coefs_guess)
    print("s_guess =>", scale_guess)
    print("coefs_actual =>", coefs_actual)
    print("scale_actual =>", scale_actual)





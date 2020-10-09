import tensorflow as tf
import imageio
import numpy as np
import matplotlib.pyplot as plt
import diffraction_functions
import datagen
import params

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
        title:str,
        plot_type:str,
        )->plt.figure:

    _coefs=sess.run(coefs,feed_dict=f)
    _scale=sess.run(scale,feed_dict=f)
    _diffraction_pattern=sess.run(diffraction_pattern,feed_dict=f)
    _obj=sess.run(obj,feed_dict=f)

    # draw the guess object
    figures=[]
    for _j in range(np.shape(diffraction_pattern_actual)[0]):
        fig=diffraction_functions.plot_amplitude_phase_meas_retreival(
                {
                    "measured_pattern":np.squeeze(diffraction_pattern_actual[_j]),
                    "tf_reconstructed_diff":np.squeeze(_diffraction_pattern[_j]),
                    "real_output":np.real(np.squeeze(_obj['beforewf'][_j])),
                    "imag_output":np.imag(np.squeeze(_obj['beforewf'][_j])),
                    "coefficients":_coefs[_j],
                    "scale":_scale[_j],
                    },
                title,
                ACTUAL=True,
                mask=True,
                plot_type=plot_type
                )
        figures.append(fig)
    return figures



if __name__ == "__main__":
    N_TESTS=28
    N=128
    datagenerator = datagen.DataGenerator(1024,N)
    coefs_actual = tf.placeholder(tf.float32, shape=[N_TESTS, len(datagenerator.zernike_cvector)])
    scale_actual = tf.placeholder(tf.float32, shape=[N_TESTS,1])
    diffraction_pattern_actual,actual_obj=make_dif_pattern(datagenerator,coefs_actual,scale_actual)

    # generate another diffraction pattern with variable
    # coefs_guess = tf.Variable(tf.truncated_normal((1,14),stddev=0.1,dtype=tf.float32))

    # start at 0
    gif_frames = [[] for _ in range(N_TESTS)]
    coefs_guess = tf.Variable(np.array(N_TESTS*[14*[0]],dtype=np.float32))
    coefs_guess = tf.sigmoid(coefs_guess)
    coefs_guess*=12
    coefs_guess-=6
    scale_guess = tf.Variable(np.array(N_TESTS*[[0]],dtype=np.float32))
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
            coefs_actual:np.array([
                                    [-6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
                                    [6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
                                    [0.0, -6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
                                    [0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
                                    [0.0, 0.0, -6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
                                    [0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
                                    [0.0, 0.0, 0.0, -6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
                                    [0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
                                    [0.0, 0.0, 0.0, 0.0, -6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
                                    [0.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
                                    [0.0, 0.0, 0.0, 0.0, 0.0, -6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
                                    [0.0, 0.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
                                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
                                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
                                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
                                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
                                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -6.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
                                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
                                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -6.0, 0.0, 0.0, 0.0, 0.0 ],
                                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0 ],
                                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -6.0, 0.0, 0.0, 0.0 ],
                                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0 ],
                                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -6.0, 0.0, 0.0 ],
                                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0 ],
                                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -6.0, 0.0 ],
                                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6.0, 0.0 ],
                                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -6.0 ],
                                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6.0 ],
                                    ]), # 28 total: 2 for each aberration
            scale_actual:np.array([
                                    [1.0],
                                    [1.0],
                                    [1.0],
                                    [1.0],
                                    [1.0],
                                    [1.0],
                                    [1.0],
                                    [1.0],
                                    [1.0],
                                    [1.0],
                                    [1.0],
                                    [1.0],
                                    [1.0],
                                    [1.0],
                                    [1.0],
                                    [1.0],
                                    [1.0],
                                    [1.0],
                                    [1.0],
                                    [1.0],
                                    [1.0],
                                    [1.0],
                                    [1.0],
                                    [1.0],
                                    [1.0],
                                    [1.0],
                                    [1.0],
                                    [1.0],
                                    ]),
                }
        error_vals={
                "diffraction_p_error":[],
                "coefs_error":[],
                "scale_error":[],
                "cost_function":[],
                }
        i_max=600
        i_skip=5
        for i in range(i_max):
            print(i,' ',end='')

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
            if i % i_skip == 0:
                print("saving image")
                _diffraction_pattern_actual=sess.run(diffraction_pattern_actual,feed_dict=f)

                figures_guess=draw_figure(sess,_diffraction_pattern_actual,coefs_guess,scale_guess,diffraction_pattern_guess,guess_obj,f,"GUESS "+str(i),plot_type='guess')
                image_guess=[]
                for fig in figures_guess:
                    fig.canvas.draw()
                    _image_guess=np.frombuffer(fig.canvas.tostring_rgb(),dtype='uint8')
                    _image_guess=_image_guess.reshape(fig.canvas.get_width_height()[::-1]+(3,))
                    plt.close(fig)
                    image_guess.append(_image_guess)

                figures_actual=draw_figure(sess,_diffraction_pattern_actual,coefs_actual,scale_actual,diffraction_pattern_actual,actual_obj,f,"ACTUAL",plot_type='original')
                image_actual=[]
                for fig in figures_actual:
                    fig.canvas.draw()
                    _image_actual=np.frombuffer(fig.canvas.tostring_rgb(),dtype='uint8')
                    _image_actual=_image_actual.reshape(fig.canvas.get_width_height()[::-1]+(3,))
                    plt.close(fig)
                    image_actual.append(_image_actual)

                # draw plot
                fig = plt.figure(figsize=(20,3))
                gs = fig.add_gridspec(1,4)
                ax=fig.add_subplot(gs[0,1:4])
                ax.plot(error_vals['diffraction_p_error'],label='Diffraction Pattern Error')
                ax.plot(error_vals['coefs_error'],label='Zernike Coefs Error')
                ax.plot(error_vals['scale_error'],label='Scale Error')
                ax.plot(error_vals['cost_function'],label='Cost Function')
                ax.set_xlim(0,i_max)
                ax.text(0.5,-0.1,"Cost Function:"+"%.4f"%sess.run(cost_function,feed_dict=f),ha='center',transform=ax.transAxes,backgroundcolor='red')
                ax.legend()

                ax = fig.add_subplot(gs[0,0])
                simulation_axes,amplitude_mask=diffraction_functions.get_amplitude_mask_and_imagesize(
                        datagenerator.N_interp,
                        int(params.params.wf_ratio*datagenerator.N_interp)
                        )
                x=simulation_axes['object']['x'] # meters
                x*=1e6
                ax.pcolormesh(x,x,amplitude_mask,cmap='gray')
                ax.set_ylabel('position [um]')

                fig.canvas.draw()
                im_plot=np.frombuffer(fig.canvas.tostring_rgb(),dtype='uint8')
                im_plot=im_plot.reshape(fig.canvas.get_width_height()[::-1]+(3,))
                plt.close(fig)
                for _gif_frames,_image_actual,_image_guess in zip(gif_frames,image_actual,image_guess):
                    image_both=np.append(_image_actual,_image_guess,axis=1)
                    image_both=np.append(im_plot,image_both,axis=0)
                    _gif_frames.append(image_both)

            # append the plots

            # train
            sess.run(train, feed_dict=f)

        for i,_gif_frames in enumerate(gif_frames):
            print('generating gif %i'%i)
            imageio.mimsave('./'+'C_test_multiple_'+str(i)+'.gif',_gif_frames,fps=3)





    print("coefs_guess =>", coefs_guess)
    print("s_guess =>", scale_guess)
    print("coefs_actual =>", coefs_actual)
    print("scale_actual =>", scale_actual)





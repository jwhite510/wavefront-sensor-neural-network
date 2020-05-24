import numpy as np
import diffraction_functions
import matplotlib.pyplot as plt
import tensorflow as tf
from PropagateTF import *


def plot_diffraction(arr,name):
    fig, ax = plt.subplots(1,2, figsize=(15,5))
    fig.suptitle(name)
    ax[0].pcolormesh(np.abs(arr))
    ax[1].pcolormesh(np.abs(np.fft.fftshift(np.fft.fft2(np.fft.fftshift(arr))))**2)

if __name__ == "__main__":
    propagate=PropagateTF()
    wavefront_before=read_complex_array("interped_arr_before")
    wavefront_before=np.expand_dims(wavefront_before,0)
    wavefront_before=np.expand_dims(wavefront_before,-1)
    with tf.Session() as sess:
        out=sess.run(propagate.through_wf,feed_dict={propagate.wavefront:wavefront_before})
        # plt.figure()
        # plt.imshow(np.squeeze(np.abs(out)))

    # propagate.through_wfs(wavefront_before)
    interped_arr_before=read_complex_array("interped_arr_before")
    plot_diffraction(interped_arr_before,"interped_arr_before")

    interped_arr_after=read_complex_array("interped_arr_after")
    plot_diffraction(interped_arr_after,"interped_arr_after")

    plot_diffraction(np.squeeze(out),"out")

    plt.show()





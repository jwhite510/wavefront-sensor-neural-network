import matplotlib.pyplot as plt
import glob
import numpy as np
import os


if __name__ == "__main__":
    run_name = "logtest"

    train_log = np.loadtxt(os.path.join(run_name, "train_log.dat"))
    validation_log = np.loadtxt(os.path.join(run_name, "validation_log.dat"))

    # training and validation error for real, imaginary, reconstruction
    plt.figure()

    plt.plot(train_log[:,1], train_log[:,2]) # real_loss
    plt.plot(train_log[:,1], train_log[:,3]) # imag_loss
    plt.plot(train_log[:,1], train_log[:,4]) # reconstruction_loss

    plt.plot(validation_log[:,1], validation_log[:,2]) # real_loss
    plt.plot(validation_log[:,1], validation_log[:,3]) # imag_loss
    plt.plot(validation_log[:,1], validation_log[:,4]) # reconstruction_loss


    measured_trace_data = {}
    for file in glob.glob(r'./{}/measured*1-0*'.format(run_name)):
        data = np.loadtxt(file)
        measured_trace_data[os.path.split(file)[-1].split(".")[0]] = data

    # plt.show()
    plt.figure()
    for m in measured_trace_data.keys():
        measured_trace_data[m]
        plt.plot(measured_trace_data[m][:,1] # epoch
                , measured_trace_data[m][:,2], # reconstruction_loss
                label=m
                )
    plt.legend()

    plt.show()





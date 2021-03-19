import matplotlib.pyplot as plt
import glob
import numpy as np
import os


def makeplots(run_name):
    train_log = np.loadtxt(os.path.join(run_name, "train_log.dat"))
    validation_log = np.loadtxt(os.path.join(run_name, "validation_log.dat"))

    # training and validation error for real, imaginary, reconstruction
    # plt.figure()
    fig, ax = plt.subplots(2,1, figsize=(5,7))
    fig.subplots_adjust(hspace=0.0, left=0.15)
    ax[0].plot(train_log[:,1], train_log[:,2]*1e3,
            label="real loss (training)",
            color="b") # real_loss

    ax[0].plot(validation_log[:,1], validation_log[:,2]*1e3,
            label="real loss(validation)",
            color="b",
            linestyle="dashed") # real_loss

    ax[0].plot(train_log[:,1], train_log[:,3]*1e3,
            label="imaginary loss (training)",
            color="r") # imag_loss


    ax[0].plot(validation_log[:,1], validation_log[:,3]*1e3,
            label="imaginary loss (validation)",
            color="r",
            linestyle="dashed") # imag_loss

    ax[0].legend()
    ax[0].set_ylabel(r"error [mse] $\cdot 10^3$")
    ax[0].set_title("Error plotted during training")

    ax[1].plot(train_log[:,1], train_log[:,4]*1e3,
            label="reconstruction loss (training)",
            color="g") # reconstruction_loss

    ax[1].plot(validation_log[:,1], validation_log[:,4]*1e3,
            label="reconstruction loss (validation)",
            color="g", linestyle="dashed") # reconstruction_loss
    ax[1].set_ylabel(r"error [mse] $\cdot 10^3$")
    ax[1].legend()
    ax[1].set_xlabel("epoch")
    fig.savefig("./training_validation_loss_{}.png".format(run_name))


    measured_trace_data = {}
    for file in glob.glob(r'./{}/*1-0*'.format(run_name)):
        data = np.loadtxt(file)
        measured_trace_data[os.path.split(file)[-1].split(".")[0]] = data

    # plt.show()
    # plt.figure()
    added_labels={}
    for m in measured_trace_data.keys():
        _label=""

        if "multiple_measurements" in m:
            fignum=3
            figtitle="multiple_measurements"
        if "z0" in m:
            fignum=4
            figtitle="z0"
        if "z-500" in m:
            fignum=5
            figtitle="z-500"
        if "z-1000" in m:
            fignum=6
            figtitle="z-1000"

        if "_None_" in m:
            color="red"
            _label="NONE"

        if "_lr_" in m:
            color="blue"
            _label="LR"

        if "_lrud_" in m:
            color="green"
            _label="LRUD"

        if "_ud_" in m:
            color="orange"
            _label="UD"

        plt.figure(fignum)
        plt.title(figtitle)
        if str(fignum)+_label in added_labels.keys():
            plt.plot(measured_trace_data[m][:,1] # epoch
                    , measured_trace_data[m][:,2]*1e3, # reconstruction_loss
                    color=color
                    )
        else:
            plt.plot(measured_trace_data[m][:,1] # epoch
                    , measured_trace_data[m][:,2]*1e3, # reconstruction_loss
                    label=_label,
                    color=color
                    )
        added_labels[str(fignum)+_label]=True
        # plt.title("Measured Trace Reconstruction Error [mse]")
        plt.xlabel("epoch")
        plt.ylabel(r"error [mse] $\cdot 10^3$")
        plt.legend()

    plt.savefig("./measured_trace_error_{}.png".format(run_name))


if __name__ == "__main__":
    makeplots("multiplefiles_5_test_A-6_0--S-0_5")
    plt.show()





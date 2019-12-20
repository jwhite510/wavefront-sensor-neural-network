import os
import matplotlib.pyplot as plt
import shutil
import numpy as np
import pandas as pd
import sys
from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def tabulate_events(dpath):
    summary_iterators = [EventAccumulator(os.path.join(dpath, dname)).Reload() for dname in os.listdir(dpath)]

    tags = summary_iterators[0].Tags()['scalars']

    for it in summary_iterators:
        assert it.Tags()['scalars'] == tags

    out = defaultdict(list)
    steps = []

    for tag in tags:
        steps = [e.step for e in summary_iterators[0].Scalars(tag)]

        for events in zip(*[acc.Scalars(tag) for acc in summary_iterators]):
            assert len(set(e.step for e in events)) == 1

            out[tag].append([e.value for e in events])

    return out, steps


def to_csv(dpath):
    dirs = os.listdir(dpath)
    # remove files that are not the tf events
    for file in dirs:
        print("file =>", file)
        # remove other files
        if 'tfevents' not in file:
            try:
                os.remove(dpath+file)
            except:
                pass
            try:
                shutil.rmtree(dpath+file)
            except:
                pass
    dirs = os.listdir(dpath)
    d, steps = tabulate_events(dpath)
    tags, values = zip(*d.items())
    np_values = np.array(values)

    for index, tag in enumerate(tags):
        df = pd.DataFrame(np_values[index], index=steps, columns=dirs)
        df.to_csv(get_file_path(dpath, tag))


def get_file_path(dpath, tag):
    file_name = tag.replace("/", "_") + '.csv'
    folder_path = os.path.join(dpath, 'csv')
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return os.path.join(folder_path, file_name)

def get_values(filename):
    with open(filename, "r") as file:
        file_contents = list(file.readlines())
        epochs = []
        error = []
        for j, line in enumerate(file_contents[1:]):
            # if j < 100:
            try:
                float(line.strip('\n').split(',')[0])
                float(line.strip('\n').split(',')[1])
                epochs.append(float(line.strip('\n').split(',')[0]))
                error.append(float(line.strip('\n').split(',')[1]))
            except:
                epochs.append(float(line.strip('\n').split(',')[0]))
                error.append(0)

        # print("len(epochs) =>", len(epochs))
        return error, epochs

def make_figure():
    run_name = sys.argv[1].split("/")[-2]

    fig = plt.figure()
    fig.text(0.5, 0.95, run_name, ha="center")


    fig.subplots_adjust(hspace=0.0, left=0.2)
    gs = fig.add_gridspec(2,1)

    # phase loss plotting
    ax = fig.add_subplot(gs[0,0])
    ax.set_title("Amplitude and Phase Retrieval Loss over Epoch")

    error, epochs = get_values("phase_loss_training.csv" )
    ax.plot(epochs, error, label="training")
    ax.set_ylabel("phase loss")
    error, epochs = get_values("phase_loss_validation.csv" )
    ax.plot(epochs, error, label="validation")
    ax.set_yscale("log")
    ax.legend()


    # amplitude loss plotting
    ax = fig.add_subplot(gs[1,0])
    error, epochs = get_values("amplitude_loss_training.csv" )
    ax.plot(epochs, error, label="training")
    ax.set_ylabel("amplitude loss")
    ax.set_xlabel("Epoch")
    error, epochs = get_values("amplitude_loss_validation.csv" )
    ax.plot(epochs, error, label="validation")
    ax.set_yscale("log")
    ax.legend()


    error, epochs = get_values("amplitude_loss_validation.csv")
    error, epochs = get_values("amplitude_loss_training.csv")

    plt.savefig(str(run_name)+'.png')
    plt.show()


if __name__ == '__main__':

    """
    specify path as the folder containing a tf.events file
    """

    # create csvs of data
    path = sys.argv[1]
    to_csv(path)


    # plot the data
    os.chdir(os.path.join(sys.argv[1], "csv"))
    make_figure()







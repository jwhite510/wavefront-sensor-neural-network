import matplotlib.pyplot as plt
import numpy as np



def convert_to_label(_phi, _x):
    fig = plt.figure(figsize=(5,10))
    gs = fig.add_gridspec(4,1)
    ax = fig.add_subplot(gs[0,0])
    ax.plot(_x, _phi)
    ax.plot([_x[49], _x[49]], [1, -1])
    ax.plot([-1, 1], [0, 0])
    ax.set_ylim([-2, 2])

    # subtract phase at center
    _phi -= _phi[49]

    ax = fig.add_subplot(gs[1,0])
    ax.plot(_x, _phi)
    ax.plot([_x[49], _x[49]], [1, -1])
    ax.plot([-1, 1], [0, 0])
    ax.set_ylim([-2, 2])
    ax.text(0.9, 0.9,"Phase Subtracted", fontsize=10, ha='center', transform=ax.transAxes, backgroundcolor="red")


    # divide by max
    _norm_factor = np.max(np.abs(_phi))
    _phi = _phi / _norm_factor
    ax = fig.add_subplot(gs[2,0])
    ax.plot(_x, _phi)
    ax.plot([_x[49], _x[49]], [1, -1])
    ax.plot([-1, 1], [0, 0])
    ax.text(0.9, 0.5,"S:"+str(_norm_factor), fontsize=10, ha='center', transform=ax.transAxes, backgroundcolor="yellow")
    ax.set_ylim([-2, 2])
    ax.text(0.9, 0.9,"divide by max", fontsize=10, ha='center', transform=ax.transAxes, backgroundcolor="red")

    # set between 0 and 1
    _phi+=1
    _phi = _phi / 2
    ax = fig.add_subplot(gs[3,0])
    ax.plot(_x, _phi)
    ax.plot([_x[49], _x[49]], [1, -1])
    ax.plot([-1, 1], [0, 0])
    ax.set_ylim([-2, 2])

    return _norm_factor, _phi

def convert_from_label(_phi, _x, _norm_factor):
    fig = plt.figure(figsize=(5,10))
    gs = fig.add_gridspec(4,1)

    # plot input
    ax = fig.add_subplot(gs[0,0])
    ax.plot(_x, _phi)
    ax.plot([_x[49], _x[49]], [1, -1])
    ax.plot([-1, 1], [0, 0])
    ax.set_ylim([-2, 2])

    # set between -1 and 1
    _phi = _phi * 2
    _phi-=1

    # plot input
    ax = fig.add_subplot(gs[1,0])
    ax.plot(_x, _phi)
    ax.plot([_x[49], _x[49]], [1, -1])
    ax.plot([-1, 1], [0, 0])
    ax.set_ylim([-2, 2])

    # multiply by factor
    _phi *= _norm_factor
    # plot input
    ax = fig.add_subplot(gs[2,0])
    ax.plot(_x, _phi)
    ax.plot([_x[49], _x[49]], [1, -1])
    ax.plot([-1, 1], [0, 0])
    ax.set_ylim([-2, 2])
    ax.text(0.9, 0.9,"Original (phase subtracted)", fontsize=10, ha='center', transform=ax.transAxes, backgroundcolor="red")


if __name__ == "__main__":

    x = np.linspace(-10, 10, 100)

    phi = np.sin(x + 10*np.random.rand()) + (2*np.random.rand()-1)*0.01*x**2 + (2*np.random.rand()-1)*0.1*x**3 + (2*np.random.rand()-1)*0.9
    norm_factor, phi = convert_to_label(phi, x)
    convert_from_label(phi, x, norm_factor)


    plt.show()


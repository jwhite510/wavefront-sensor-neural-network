import matplotlib.pyplot as plt
import numpy as np



def convert_to_label(_phi, _x):
    fig = plt.figure(figsize=(5,10))
    gs = fig.add_gridspec(4,1)
    fig.text(0.5, 0.95, "convert to label", ha="center", size=30)
    ax = fig.add_subplot(gs[0,0])
    ax.text(0.0, 0.9,"(1)", fontsize=10, ha='center', transform=ax.transAxes, backgroundcolor="yellow")
    ax.plot(_x, _phi)
    ax.plot([_x[49], _x[49]], [1, -1], "b--")
    ax.plot([-1, 1], [0, 0], "b--")
    ax.set_ylim([-2, 2])

    # subtract phase at center
    _phi -= _phi[49]

    ax = fig.add_subplot(gs[1,0])
    ax.text(0.0, 0.9,"(2)", fontsize=10, ha='center', transform=ax.transAxes, backgroundcolor="yellow")
    ax.plot(_x, _phi)
    ax.plot([_x[49], _x[49]], [1, -1], "b--")
    ax.plot([-1, 1], [0, 0], "b--")
    ax.set_ylim([-2, 2])
    ax.text(0.9, 0.9,"Phase Subtracted", fontsize=10, ha='center', transform=ax.transAxes, backgroundcolor="red")


    # divide by max
    _norm_factor = np.max(np.abs(_phi))
    _phi = _phi / _norm_factor
    ax = fig.add_subplot(gs[2,0])
    ax.text(0.0, 0.9,"(3)", fontsize=10, ha='center', transform=ax.transAxes, backgroundcolor="yellow")
    ax.plot(_x, _phi)
    ax.plot([_x[49], _x[49]], [1, -1], "b--")
    ax.plot([-1, 1], [0, 0], "b--")
    ax.text(0.9, 0.1,"S:"+str(_norm_factor), fontsize=10, ha='center', transform=ax.transAxes, backgroundcolor="yellow")
    ax.set_ylim([-2, 2])
    ax.text(0.9, 0.9,r"divide by max($\|\phi\|$)", fontsize=10, ha='center', transform=ax.transAxes, backgroundcolor="red")
    ax.text(0.1, 0.1,r"now $\phi$ is between -1 and 1", fontsize=10, ha='center', transform=ax.transAxes, backgroundcolor="red")

    # set between 0 and 1
    _phi+=1
    _phi = _phi / 2
    ax = fig.add_subplot(gs[3,0])
    ax.text(0.0, 0.9,"(4)", fontsize=10, ha='center', transform=ax.transAxes, backgroundcolor="yellow")
    ax.plot(_x, _phi)
    ax.plot([_x[49], _x[49]], [1, -1], "b--")
    ax.plot([-1, 1], [0, 0], "b--")
    ax.set_ylim([-2, 2])
    ax.text(0.9, 0.9,"set between 0 and 1", fontsize=10, ha='center', transform=ax.transAxes, backgroundcolor="red")

    return _norm_factor, _phi

def convert_from_label(_phi, _x, _norm_factor):
    fig = plt.figure(figsize=(5,10))
    gs = fig.add_gridspec(4,1)
    fig.text(0.5, 0.95, "convert from label", ha="center", size=30)

    # plot input
    ax = fig.add_subplot(gs[0,0])
    ax.text(0.0, 0.9,"(1)", fontsize=10, ha='center', transform=ax.transAxes, backgroundcolor="yellow")
    ax.plot(_x, _phi)
    ax.plot([_x[49], _x[49]], [1, -1], "b--")
    ax.plot([-1, 1], [0, 0], "b--")
    ax.set_ylim([-2, 2])

    # set between -1 and 1
    _phi = _phi * 2
    _phi-=1

    # plot input
    ax = fig.add_subplot(gs[1,0])
    ax.text(0.0, 0.9,"(2)", fontsize=10, ha='center', transform=ax.transAxes, backgroundcolor="yellow")
    ax.plot(_x, _phi)
    ax.plot([_x[49], _x[49]], [1, -1], "b--")
    ax.plot([-1, 1], [0, 0], "b--")
    ax.set_ylim([-2, 2])

    # multiply by factor
    _phi *= _norm_factor
    # plot input
    ax = fig.add_subplot(gs[2,0])
    ax.text(0.0, 0.9,"(3)", fontsize=10, ha='center', transform=ax.transAxes, backgroundcolor="yellow")
    ax.plot(_x, _phi)
    ax.plot([_x[49], _x[49]], [1, -1], "b--")
    ax.plot([-1, 1], [0, 0], "b--")
    ax.set_ylim([-2, 2])
    ax.text(0.9, 0.9,"Original (phase subtracted)", fontsize=10, ha='center', transform=ax.transAxes, backgroundcolor="red")

    return _phi


if __name__ == "__main__":

    x = np.linspace(-10, 10, 100)

    phi = np.sin(x + 10*np.random.rand()) + (2*np.random.rand()-1)*0.01*x**2 + (2*np.random.rand()-1)*0.001*x**3 + (2*np.random.rand()-1)*0.9
    # phi = np.sin(x + 10*np.random.rand())

    phi_original = np.array(phi)
    phi_original -= phi_original[49]
    norm_factor, phi = convert_to_label(phi, x)
    phi_converted = convert_from_label(phi, x, norm_factor)

    plt.figure(3)
    plt.plot(x, phi_original, "r")
    plt.plot(x, phi_converted, "b--")
    plt.plot(x, np.abs(phi_converted-phi_original), "c")
    plt.plot([x[49],x[49]], [-1,1], "c")






    plt.show()


import numpy as np
import matplotlib
matplotlib.use('TkAgg') # fixme
import matplotlib.pyplot as plt
import time


palette = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33','#a65628','#f781bf','#999999']


def augment(x):
    return np.concatenate((x, [1]))


def plot_errors(errors, show=True):
    plt.plot(errors)

    if show:  plt.show()


def plot_dots(inputs, targets=None, s=100, show=True):
    if targets is None:
        plt.scatter(inputs[:,0], inputs[:,1], s=s, c=palette[-1])
    else:
        for i, c in enumerate(set(targets)):
            plt.scatter(inputs[targets==c,0], inputs[targets==c, 1], s=s, c=palette[i])

    if show:  plt.show()


def limits(values, gap=0.05):
    x0 = np.min(values)
    x1 = np.max(values)
    xg = (x1 - x0) * gap
    return np.array((x0-xg, x1+xg))


def plot_decision(w, inputs, targets=None, s=100, show=True):
    plot_dots(inputs, targets, s=s, show=False)

    X = limits(inputs[:,0])
    Y = - ( w[0] * X + w[2] ) / w[1];
    plt.plot(X, Y, color='black')

    plt.xlim(limits(inputs[:,0]))
    plt.ylim(limits(inputs[:,1]))

    if show:  plt.show()


## interactive drawing, very fragile....

def clear():
    plt.clf()


def ion():
    plt.ion()
    plt.show()
    time.sleep(0.1)


def ioff():
    plt.ioff()
    plt.close()


def redraw():
    plt.gcf().canvas.draw()
    time.sleep(0.1)

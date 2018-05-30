import numpy as np

from util import *


class Perceptron():

    def __init__(self, dim):
        self.dim     = dim
        self.weights =  np.random.random(size = dim + 1) # FIXME


    def train(self, inputs, targets, alpha=0.1, eps=20, trace=False):
        (count, _) = inputs.shape

        if trace:
            plot_decision(self.weights, inputs, targets, show=False)
            ion()

        errors = []

        for ep in range(eps):
            print('Ep {:3d}/{}:'.format(ep+1, eps), end='')
            E = 0

            for i in np.random.permutation(count):
                x = augment(inputs[i]) # FIXME
                d = targets[i]          # FIXME

                net = np.dot(self.weights, x)  # FIXME
                if net >= 0:
                    y = 1          # FIXME
                else:
                    y = 0

                e = d - y # FIXME
                E += e**2 / 2.0 # Python 2 has C-like / semantics, Python 3 has // for integer division, / for float

                self.weights += alpha* e *x# FIXME

            errors.append(E)

            print('E = {:.3f}'.format(E))

            if trace:
                clear()
                plot_decision(self.weights, inputs, targets, show=False)
                redraw()

        if trace:
            ioff()

        return errors

import atexit
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


## utility

def augment(X):
    if X.ndim == 1:
        return np.concatenate((X, [1]))
    else:
        pad = np.ones((1, X.shape[1]))
        return np.concatenate((X, pad), axis=0)


def onehot_decode(X):
    return np.argmax(X, axis=0)

## plotting


def plot_errors(title, errors, test_error=None, block=True):
    plt.figure()

    plt.plot(errors)

    if test_error:
        plt.plot([test_error]*len(errors))

    plt.tight_layout()
    plt.gcf().canvas.set_window_title(title)
    
    plt.savefig('errors.png')
    plt.show(block=block)


def plot_reg_density(title, inputs, targets, outputs=None, s=70, block=True):

    plt.figure(figsize=(9,9))

    if outputs is not None:
        plt.subplot(2,1,2)
        plt.title('Predicted')
        plt.scatter(inputs[0], inputs[1], s=s*outputs)

        plt.subplot(2,1,1)
        plt.title('Original')

    plt.scatter(inputs[0], inputs[1], s=s*targets)
    plt.gcf().canvas.set_window_title(title)
    plt.tight_layout()
    plt.savefig('reg_density.png')
    plt.show(block=block)


## non-blocking figures still block at end

def finish():
    plt.show(block=True) # block until all figures are closed


atexit.register(finish)

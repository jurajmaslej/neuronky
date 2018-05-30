import atexit
import numpy as np
import matplotlib
matplotlib.use('TkAgg') # todo: remove or change if not working
import matplotlib.pyplot as plt
import time


## utility

def augment(X):
    if X.ndim == 1:
        return np.concatenate((X, [1]))
    else:
        pad = np.ones((1, X.shape[1]))
        return np.concatenate((X, pad), axis=0)

def onehot_decode(X):
    return np.argmax(X, axis=0)

def onehot_encode(L, c):
    if isinstance(L, int):
        L = [L]
    n = len(L)
    out = np.zeros((c, n))
    #print(out.shape)
    out[L, range(n)] = 1
    return np.squeeze(out)


## plotting

palette = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33','#a65628','#f781bf','#999999']


def limits(values, gap=0.05):
    x0 = np.min(values)
    x1 = np.max(values)
    xg = (x1 - x0) * gap
    return np.array((x0-xg, x1+xg))


def plot_errors(title, errors, test_error=None, block=True):
    plt.figure(1)
    plt.clf()

    plt.plot(errors)

    if test_error:
        plt.plot([test_error]*len(errors))

    plt.tight_layout()
    plt.gcf().canvas.set_window_title(title)
    plt.show(block=block)


def plot_both_errors(trainCEs, trainREs, testCE=None, testRE=None, pad=None, block=True):
    plt.figure(2)
    plt.clf()

    if pad is None:
        pad = max(len(trainCEs), len(trainREs))
    else:
        trainCEs = np.concatentate((trainCEs, [None]*(pad-len(trainCEs))))
        trainREs = np.concatentate((trainREs, [None]*(pad-len(trainREs))))

    plt.subplot(2,1,1)
    plt.title('Classification accuracy')
    plt.plot(100*np.array(trainCEs))
    
    if testCE is not None:
        plt.plot([100*testCE]*pad)

    plt.subplot(2,1,2)
    plt.title('Model loss (MSE)')
    plt.plot(trainREs)

    if testRE is not None:
        plt.plot([testRE]*pad)

    plt.tight_layout()
    plt.gcf().canvas.set_window_title('Errors')
    plt.show(block=block)


def plot_dots(inputs, labels=None, predicted=None, test_inputs=None, test_labels=None, test_predicted=None, s=60, i_x=0, i_y=1, block=True):
    plt.figure(3)
    plt.clf()

    if labels is None:
        plt.gcf().canvas.set_window_title('Data distribution')
        plt.scatter(inputs[i_x,:], inputs[i_y,:], s=s, c=palette[-1], edgecolors=[0.4]*3, alpha=0.5)

    elif predicted is None:
        plt.gcf().canvas.set_window_title('Class distribution')
        for i, c in enumerate(set(labels)):
            plt.scatter(inputs[i_x,labels==c], inputs[i_y,labels==c], s=s, c=palette[i], edgecolors=[0.4]*3)

    else:
        plt.gcf().canvas.set_window_title('Predicted vs. actual')
        for i, c in enumerate(set(labels)):
            plt.scatter(inputs[i_x,labels==c], inputs[i_y,labels==c], s=3.0*s, c=palette[i], edgecolors=None, alpha=0.5)

        for i, c in enumerate(set(labels)):
            plt.scatter(inputs[i_x,predicted==c], inputs[i_y,predicted==c], s=0.5*s, c=palette[i], edgecolors=None)

    if test_inputs is not None:
        if test_labels is None:
            plt.scatter(test_inputs[i_x,:], test_inputs[i_y,:], marker='s', s=s, c=palette[-1], edgecolors=[0.4]*3, alpha=0.5)

        elif test_predicted is None:
            for i, c in enumerate(set(test_labels)):
                plt.scatter(test_inputs[i_x,test_labels==c], test_inputs[i_y,test_labels==c], marker='s', s=s, c=palette[i], edgecolors=[0.4]*3)

        else:
            for i, c in enumerate(set(test_labels)):
                plt.scatter(test_inputs[i_x,test_labels==c], test_inputs[i_y,test_labels==c], marker='s', s=3.0*s, c=palette[i], edgecolors=None, alpha=0.5)

            for i, c in enumerate(set(test_labels)):
                plt.scatter(test_inputs[i_x,test_predicted==c], test_inputs[i_y,test_predicted==c], marker='s', s=0.5*s, c=palette[i], edgecolors=None)

    plt.xlim(limits(inputs[i_x,:]))
    plt.ylim(limits(inputs[i_y,:]))
    plt.tight_layout()
    plt.show(block=block)


def plot_areas(model, inputs, labels=None, w=30, h=20, i_x=0, i_y=1, block=True):
    plt.figure(4)
    plt.clf()
    plt.gcf().canvas.set_window_title('Decision areas')

    dim = inputs.shape[0]
    data = np.zeros((dim, w*h))

    # # "proper":
    # X = np.linspace(*limits(inputs[i_x,:]), w)
    # Y = np.linspace(*limits(inputs[i_y,:]), h)
    # YY, XX = np.meshgrid(Y, X)
    #
    # for i in range(dim):
    #     data[i,:] = np.mean(inputs[i,:])
    # data[i_x,:] = XX.flat
    # data[i_y,:] = YY.flat

    X1 = np.linspace(*limits(inputs[0,:]), w)
    Y1 = np.linspace(*limits(inputs[1,:]), h)
    X2 = np.linspace(*limits(inputs[2,:]), w)
    Y2 = np.linspace(*limits(inputs[3,:]), h)
    YY1, XX1 = np.meshgrid(Y1, X1)
    YY2, XX2 = np.meshgrid(Y2, X2)
    data[0,:] = XX1.flat
    data[1,:] = YY1.flat
    data[2,:] = XX2.flat
    data[3,:] = YY2.flat

    outputs, *_ = model.forward(data)
    outputs = outputs.reshape((-1,w,h))

    outputs -= np.min(outputs, axis=0, keepdims=True)
    outputs  = np.exp(1*outputs)
    outputs /= np.sum(outputs, axis=0, keepdims=True)

    plt.imshow(outputs.T)

    plt.tight_layout()
    plt.show(block=block)


## interactive drawing, very fragile....

wait = 0.0

def clear():
    plt.clf()


def ion():
    plt.ion()
    time.sleep(wait)


def ioff():
    plt.ioff()


def redraw():
    plt.gcf().canvas.draw()
    time.sleep(wait)


## non-blocking figures still block at end

def finish():
    plt.show(block=True) # block until all figures are closed


atexit.register(finish)

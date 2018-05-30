import atexit
import numpy as np
import matplotlib
matplotlib.use('TkAgg') # fixme
import matplotlib.pyplot as plt

import noise


def show_images(inputs, title=None, block=False):
    print("plot")
    print(np.shape(inputs))
    images = inputs.reshape(noise.img_shape)
    (_, _, count) = images.shape

    plt.figure()
    h, w = grid_size(count)

    if title:
        plt.gcf().canvas.set_window_title(title)

    for i in range(count):
        plt.subplot(h,w,i+1)
        plt.imshow(images[:,:,i].T, interpolation='nearest', cmap='gray', vmin=0, vmax=255)
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout()
    plt.show(block=block)


def grid_size(count, grid_aspect=1.3, img_aspect=1):
    cols = 1
    rows = 1

    while count > rows*cols:
        if (cols/rows) > (grid_aspect * img_aspect):
            rows += 1
        else:
            cols += 1

    return rows, cols


def finish():
    plt.show(block=True) # block until all figures are closed


atexit.register(finish)

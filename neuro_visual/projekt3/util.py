import atexit
import numpy as np
import matplotlib
matplotlib.use('TkAgg') # todo: remove or change if not working
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import os
import time


## globals

width  = None
height = None

def util_setup(w, h):
    global width, height
    width  = w
    height = h


## plotting

def plot_state(s, errors=None, index=None, max_eps=None, rows=1, row=1, size=2, aspect=2, title=None, block=True):
    if plot_state.fig is None:
        plot_state.fig = plt.figure(figsize=(size,size*rows) if errors is None else ((1+aspect)*size,size*rows))
        plot_state.fig.canvas.mpl_connect('key_press_event', keypress)

        gs = gridspec.GridSpec(rows, 2, width_ratios=[1, aspect])
        plot_state.grid = {(r,c): plt.subplot(gs[r,c]) for r in range(rows) for c in range(2 if errors else 1)}

        plt.subplots_adjust()
        plt.tight_layout()

    plot_state.fig.show() # foreground, swith plt.(g)cf

    ax = plot_state.grid[row-1,0]
    ax.clear()
    ax.imshow(s.reshape((height, width)), cmap='gray', interpolation='nearest', vmin=-1, vmax=+1)
    ax.set_xticks([])
    ax.set_yticks([])

    if index:
        ax.scatter(index%width, index//width, s=150)

    if errors is not None:
        ax = plot_state.grid[row-1,1]
        ax.clear()
        ax.plot(errors)

        if max_eps:
            ax.set_xlim(0, max_eps-1)

        ylim = ax.get_ylim()
        ax.vlines(np.arange(0, len(errors), width*height)[1:], ymin=ylim[0], ymax=ylim[1], color=[0.8]*3, lw=1)
        ax.set_ylim(ylim)

    plt.gcf().canvas.set_window_title(title or 'State')
    plt.show(block=block)

plot_state.fig = None


def plot_states(S, E=None, P=None, title=None, block=True):
    plt.figure(2, figsize=(9,3)).canvas.mpl_connect('key_press_event', keypress)
    plt.clf()

    for i, s in enumerate(S):
        
        plt.subplot(1, len(S), i+1)
        plt.imshow(s.reshape((height, width)), cmap='gray', interpolation='nearest', vmin=-1, vmax=+1)
        if E:
            plt.title('energy: ' + str(E[i]), fontsize = 9)
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout()
    plt.gcf().canvas.set_window_title(title or 'States')
    #plt.show(block=block)
    plt.savefig(title)

def plot_stats( overlap, S, E, letter = None, noise = None, title = None):
	plt.figure()
	#print('plt sts overlap len ', len(overlap))
	labels = ['A','B','X','O']
	for overl_letter, lab in zip(overlap, labels):
		index = range(0,len(overl_letter))
		plt.plot(index, overl_letter, label= lab)
	plt.legend()
	if title:
		fname = title
	else:
		fname = letter + str(noise) + '.png'
	plt.savefig(fname)
	#plt.show()
	plt.close()
	
def plot_atractor_stats(values):
	plt.figure()
	x = np.arange(len(values))
	plt.bar(x, values)
	plt.xticks(x, ('True', 'Spurious', 'Cycled'))
	plt.savefig('atractor_stats.png')
	plt.close()

## interactive drawing, very fragile....

wait = 0.01

def clear():
    plt.clf()


def ion():
    plt.ion()
    time.sleep(wait)


def ioff():
    plt.ioff()


def redraw():
    plt.gcf().canvas.draw()
    plt.waitforbuttonpress(timeout=0.001)
    # time.sleep(wait)

def keypress(e):
    if e.key in {'q', 'escape'}:
        os._exit(0) # unclean exit, but exit() or sys.exit() won't work

    if e.key in {' ', 'enter'}:
        plt.close() # skip blocking figures


## non-blocking figures still block at end

def finish():
    #plt.show(block=True) # block until all figures are closed
    pass

atexit.register(finish)

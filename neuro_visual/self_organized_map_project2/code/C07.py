import numpy as np
import random

from som import *
import matplotlib.pyplot as plt
from data_loader import Data_loader



def L_1(X, Y):
    d = np.abs(X[0] - Y[0]) + np.abs(X[1] - Y[1])
    return d # TODO

def L_2(X, Y):
    #print(X.shape)
    #print(X)
    #print(Y)
    d = np.sqrt(((X[0] - Y[0])**2 + (X[1] - Y[1])**2)) 
    return d # TODO

def L_max(X, Y):
    x = np.abs(X[0] - Y[0])
    y = np.abs(X[1] - Y[1])
    d = np.max((x, y))
    return d # TODO

def close_plot():
	try:
		plt.close()
	except:
		pass

def plot_neuron_classes(model):
	close_plot()
	#plt.xlabel('x position')
	#plt.ylabel('y position')
	plt.imshow(model.class_label)
	plt.show()
	
def plot_err(avg_quant_err, label):
	close_plot()
	a =  [i for i in range(0, len(avg_quant_err))]
	plt.plot(a, avg_quant_err, '-')
	plt.xlabel('generation')
	plt.ylabel(label)
	graph_name = label + '.png'
	plt.savefig(graph_name)
	plt.close()

def plot_count(model, graph_name):
	close_plot()

	x,y, clr_list, size_list = [], [], [], []
	colors=["red", "black", "blue"]
	
	for i in range(0,len(model.count_activations)):
		for j in range(0, len(model.count_activations[:,0])):
			#if model.count_activations[i,j] != 0:
			x.append(i)
			y.append(j)
			clr_list.append(colors[ int(model.class_label[i,j]) - 1])
	plt.xlabel('x position')
	plt.ylabel('y position')
	plt.scatter(x, y, s = model.count_activations*18, c = clr_list)
	plt.grid()
	plt.savefig(graph_name + '.png')
	
def plot_heatmap(model, graph_name):
	close_plot()
	for i in range(0,7):
		close_plot()
		f = plt.figure('heatmap')
		plt.imshow(model.weights[:,:,i])
		plt.colorbar()
		f.savefig(graph_name + '_' + str(i) + '_' + '.png')
		
def plot_umatrix(model, graph_name):
	close_plot()
	
	horizontal = []
	horizon_all = []
	f = plt.figure('horizontal')
	for i in range(0,len(model.weights) - 1):
		horizontal = []
		for j in range(0, len(model.weights) - 1):
			dst = np.sum(model.weights[i,j,:] - model.weights[i + 1,j,:])
			horizontal.append(dst)
		horizon_all.append(horizontal)
	plt.imshow(horizon_all, cmap = plt.cm.gray)
	plt.colorbar()
	f.savefig(graph_name + '_horizontal_' + '.png')
	
	vertical = []		#in fact vertical
	vertical_all = []	#vertical
	f = plt.figure('vertical')
	for i in range(0,len(model.weights) - 1):
		vertical = []
		for j in range(0, len(model.weights) - 1):
			dst = np.sum(model.weights[i,j,:] - model.weights[i,j + 1,:])
			vertical.append(dst)
		vertical_all.append(vertical)
	plt.imshow(vertical_all, cmap= plt.cm.gray)
	plt.colorbar()
	f.savefig(graph_name + '_vertical_' + '.png')
	
## load data

d = Data_loader('dataset.txt')
inputs = d.dataset.T
(dim, count) = inputs.shape

## train model

rows = 30
cols = 30  

metric = L_max

top_left = np.array((0, 0))
bottom_right = np.array((rows-1, cols-1))
lambda_s = metric(top_left, bottom_right) *0.5# there was *0.5
model = SOM(dim, rows, cols, inputs)
model.train(inputs, discrete=False, metric=metric, alpha_s=0.7, alpha_f=0.01, lambda_s=lambda_s,
            lambda_f=0.1, eps=100, in3d=False)

plot_err(model.avg_quant_err, 'avg_quant_err')
plot_err(model.avg_adjustment, 'avg_adjustment')
plot_count(model, 'counts')
plot_heatmap(model, 'heatmap')
plot_umatrix(model, 'umatrix')
plot_neuron_classes(model)

import numpy as np

from regressor import *
from util import *


## load data

data = np.loadtxt('mhd-easy.dat')

data = data[:, data[-1] != 0] # prune empty cells

inputs  = data[0:2]
targets = data[2:]  # keep the regression targets as a 2D-matrix with 1 column

# plot_reg_density('Density', inputs, targets)


## normalize inputs

inputs -= np.mean(inputs)
inputs /= np.std(inputs)

targets -= np.mean(targets)
targets /= np.std(targets)


## train & visualize
print('shapes')
print(inputs.shape[0])
print(targets.shape[0])

model = MLPRegressor(inputs.shape[0], 20, targets.shape[0])
trainREs = model.train(inputs, targets, alpha=0.05, eps=100)

outputs = model.predict(inputs)

plot_reg_density('Density', inputs, targets, outputs, block=False)
plot_errors('Model loss', trainREs, block=False)

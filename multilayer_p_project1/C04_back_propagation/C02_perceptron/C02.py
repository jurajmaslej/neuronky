import numpy as np

from perceptron import *
from util import *


data = np.loadtxt('linsep.dat') # TOOD: and, or, xor, linsep
assert(data.ndim == 2)

inputs  = data[:,:-1]
targets = data[:,-1].astype(int)

# plot_dots(inputs, targets)

(count, dim) = inputs.shape

model = Perceptron(dim)
errors = model.train(inputs, targets, trace=True)

plot_decision(model.weights, inputs, targets)

plot_errors(errors)

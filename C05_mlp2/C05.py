import numpy as np
import random

from classifier import *
from util import *


## load data

data = np.loadtxt('iris.dat').T

inputs = data[:-1]
labels = data[-1].astype(int) - 1 # last column is class, starts from 1

(dim, count) = inputs.shape


## split training & test set

# maybe:
#ind = data # TODO
#random.shuffle(ind)
split = int(count * 0.8) # TODO
#print(ind.shape)
#train_ind = ind[:,:split] # TODO
#test_ind  = ind[:,split:] # TODO
#print(train_ind.shape)
#print(test_ind.shape)
#print(ind[0])
#print(train_ind.T)
# /maybe
#print(data.T[0])
idx = np.random.randint(count, size = count)
train_ind = idx[:split]
#print (train_ind)
test_ind = idx[split:]
train_inputs = data.T[train_ind,:-1] # TODO
train_labels = data.T[train_ind,-1].astype(int) # TODO
#print(train_inputs.shape)
#print("train label ",  train_labels[0])
train_labels = np.atleast_2d(train_labels).T
#print(train_labels)
#print(train_inputs[0])

test_inputs =  data.T[test_ind,:-1] # TODO
test_labels = data.T[test_ind,-1].astype(int) # TODO
#print("train l sh ", train_labels.shape)
train_labels -= 1
test_labels -= 1
#train_inputs = augment(train_inputs)
#train_labels = augment(train_labels).astype(int)
#print(test_inputs.shape)
#print(test_labels[0])
#print(test_inputs[0])

# plot_dots(train_inputs, test_inputs=test_inputs)
# plot_dots(train_inputs, train_labels, None, test_inputs, test_labels, None)


## train & test model

model = MLPClassifier(dim, 20, np.max(labels)+1)
trainCEs, trainREs = model.train(train_inputs, train_labels, alpha=0.05, eps=2, trace=True, trace_interval=100)

testCE, testRE = model.test(test_inputs, test_labels)
print('Final testing error: CE = {:6.2%}, RE = {:.5f}'.format(testCE, testRE))

train_predicted = model.predict(train_inputs)
test_predicted  = model.predict(test_inputs)

# plot_dots(train_inputs, train_labels, train_predicted, test_inputs, test_labels, test_predicted, block=False)
plot_dots(train_inputs, None, None, test_inputs, test_labels, test_predicted, block=False)
plot_both_errors(trainCEs, trainREs, testCE, testRE, block=False)

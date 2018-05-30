import numpy as np

from util import *


class LinearAutoassociator():

    def __init__(self, dim, count):
        self.count   = count
        self.dim     = dim
        self.weights = 0 # FIXME

    def reconstruct(self, input):

        print("vahy")
        print(np.shape(self.weights))
        print("input")
        print(np.shape(input))

        return np.dot(self.weights, input)


    def novelty(self, input):
        return np.dot(self.weights - np.identity(self.dim), input)
        #return self.weights - np.identity(self.dim)


    def analytical(self, inputs):
        if self.dim < self.count:
            self.weights = np.dot(inputs, np.dot(np.transpose(inputs), np.linalg.inv(np.dot(inputs, np.transpose(inputs)))))
        else:
            self.weights = np.dot(inputs, np.dot(np.linalg.inv(np.dot(np.transpose(inputs), inputs)), np.transpose(inputs)))

        #print("pinv")
        #print(np.shape(np.dot(inputs, np.linalg.pinv(inputs))))
        #print("")
        #print(np.shape(self.weights))

    def iterative(self, inputs):
        (_, count) = inputs.shape
        print("count")
        print(count)

        for i in range(count):
            x = inputs[:, i]
            z = x - np.dot(self.weights, np.transpose(x))
            self.weights += np.outer(z, z)/(np.linalg.norm(z)**2)
            # ...

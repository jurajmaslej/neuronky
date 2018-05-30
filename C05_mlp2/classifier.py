import numpy as np

from mlp import *
from util import *


class MLPClassifier(MLP):

    def __init__(self, dim_in, dim_hid, n_classes):
        self.n_classes = n_classes
        super().__init__(dim_in, dim_hid, dim_out=n_classes)

    
    ## functions

    def cost(self, targets, outputs): # new
        return np.sum((targets - outputs)**2, axis=0)
    
    def f_hid(self, x): # override
        return(1/(1 + np.exp(-x))) # sigmoid

    def df_hid(self, x): # override

        return self.f_hid(x)*(1 - self.f_hid(x)) # derivation of sigmoid

    def f_out(self, x): # override
        return x # linear

    def df_out(self, x): # override
        return 1 # derivation of linear


    ## prediction pass

    def predict(self, inputs):
        print()
        outputs, *_ = self.forward(inputs)  # if self.forward() can take a whole batch
        # outputs = np.stack([self.forward(x)[0] for x in inputs.T]) # otherwise
        return onehot_decode(outputs)


    ## testing pass

    def test(self, inputs, labels):
        outputs = 0 # FIXME
        targets = 0 # FIXME
        predicted = 0 # FIXME 
        CE = 0 # FIXME
        RE = 0 # FIXME
        return CE, RE


    ## training

    def train(self, inputs, labels, alpha=0.1, eps=100, trace=False, trace_interval=10):
        (_, count) = inputs.shape
        targets = onehot_encode(labels, self.n_classes)
        targets= self.my_encode(labels)
        #print(targets)

        if trace:
            ion()

        CEs = []
        REs = []

        for ep in range(eps):
            print('Ep {:3d}/{}: '.format(ep+1, eps), end='')
            CE = 0
            RE = 0

            for i in np.random.permutation(count):
                x = inputs[i,:] # FIXME
                d = targets[i,:] # FIXME
                #print("d sh ", d.shape)
                print ('###',augment(x).shape)
                y, dW_hid, dW_out = self.backward(x, d)
                #print (onehot_decode(y))

                #CE += labels[i] != onehot_decode(y)
                CE += labels[i] != self.my_decode(y)
                RE += self.cost(d,y)

                self.W_hid += alpha * dW_hid # FIXME
                self.W_out += alpha * dW_out # FIXME
            #print(CE, count)
            #CE /= count
            #RE /= count

            #CEs.append(CE)
            #REs.append(RE)

            #print('CE = {:6.2%}, RE = {:.5f}'.format(CE, RE))

            if trace and ((ep+1) % trace_interval == 0):
                predicted = self.predict(inputs)
                plot_dots(inputs, labels, predicted, block=False)
                plot_both_errors(CEs, REs, block=False)
                redraw()

        if trace:
            ioff()

        print()

        return CEs, REs

    def my_encode(self, targets):
        outputs = np.array([0,0,0])
        for i in targets:
            if i[0] == 0:
                #print (outputs.shape)
                outputs = np.vstack((outputs, np.array([1,0,0])))
            if i[0] == 1:
                outputs = np.vstack((outputs, np.array([0,1,0])))
            if i[0] == 2:
                outputs = np.vstack((outputs, np.array([0,0,1])))
        
        outputs = outputs[1:,:]
        return outputs
        
    def my_decode(self, targets):
        if targets[0] == 1:
            return 0
        if targets[1] == 1:
            return 1
        if targets[2] == 1:
            return 2
import numpy as np

from mlp import *
from util import *


class MLPRegressor(MLP):

    def __init__(self, dim_in, dim_hid, dim_out):
        super().__init__(dim_in, dim_hid, dim_out)

    
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
        outputs, *_ = self.forward(inputs)  # if self.forward() can take a whole batch
        # outputs = np.stack([self.forward(x)[0] for x in inputs.T]) # otherwise
        return outputs


    ## training

    def train(self, inputs, targets, alpha=0.1, eps=100):
        (_, count) = inputs.shape

        errors = []
        print('inputs sh', targets.shape)
        for ep in range(eps):
            print('Ep {:3d}/{}: '.format(ep+1, eps), end='')
            E = 0

            for i in np.random.permutation(count):
                x = inputs[:, i] # FIXME
                #print (x)
                #print ("x sh ", x.shape)
                d = targets[:, i] # FIXME
                #print(d)

                y, dW_hid, dW_out = self.backward(x, d)
                #print('dw hid shape ', dW_hid.shape)
                E += self.cost(d,y)

                self.W_hid += alpha * dW_hid # FIXME
                self.W_out += alpha * dW_out # FIXME

            E /= count
            errors.append(E)
            print('E = {:.3f}'.format(E))

        return errors

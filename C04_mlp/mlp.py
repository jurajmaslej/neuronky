import numpy as np

from util import *


## Multi-Layer Perceptron
# (abstract base class)

class MLP():

    def __init__(self, dim_in, dim_hid, dim_out):
        self.dim_in     = dim_in
        self.dim_hid    = dim_hid
        self.dim_out    = dim_out

        self.W_hid = np.random.randn(dim_hid, dim_in + 1) # FIXME
        self.W_out = np.random.randn(dim_out, dim_hid + 1) # FIXME
        print ('whid shape', self.W_hid.shape)
        print ('wout shape', self.W_out.shape)

    ## activation functions & derivations
    # (not implemented, to be overriden in derived classes)

    def f_hid(self, x):
        pass 

    def df_hid(self, x):
        pass 

    def f_out(self, x):
        pass 

    def df_out(self, x):
        pass 


    ## forward pass
    # (single input vector)

    def forward(self, x):
        #print(x.shape)
        a = np.dot(self.W_hid, augment(x)) # FIXME
        #print(a.shape)
        h = self.f_hid(a) # FIXME
        b = np.dot(self.W_out, augment(h)) # FIXME
        y = self.f_out(b) # FIXME

        return y, b, h, a


    ## forward & backprop pass
    # (single input and target vector)

    def backward(self, x, d):
        y, b, h, a = self.forward(x)

        g_out = (d - y) * self.df_out(b) # FIXME
        #print ('gout shape ' , g_out.shape)
        #print ('wout shape', self.W_out.shape)
        g_hid = np.dot(self.W_out.T[:-1], g_out) * self.df_hid(a) # FIXME

        dW_out = np.outer(g_out, augment(h)) # FIXME
        dW_hid = np.outer(g_hid, augment(x)) # FIXME

        return y, dW_hid, dW_out

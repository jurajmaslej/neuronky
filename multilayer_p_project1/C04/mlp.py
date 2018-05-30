import numpy as np

from util import *
import math


## Multi-Layer Perceptron
# (abstract base class)

class MLP():

    def __init__(self, dim_in, dim_hid, dim_out):
        self.dim_in     = dim_in
        self.dim_hid    = dim_hid
        self.dim_out    = dim_out

        self.W_hid = np.random.rand(self.dim_hid, self.dim_in) # FIXME
        self.W_out = np.random.rand(self.dim_out, self.dim_hid) # FIXME
        


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
        a = np.dot(self.W_hid, x) # FIXME
        h = self.f_hid(a) # FIXME
        #print("wout shape ", self.W_out.shape)
        #print("h shape ", h.shape)
        b = np.dot(self.W_out, h)# FIXME
        #b = np.outer(self.W_out, h)
        #print ("b sh ", b.shape)
        y = self.f_out(b) # FIXME

        return y, b, h, a


    ## forward & backprop pass
    # (single input and target vector)
    def compute_ghid(self, g_out, a):
        #print ("wout sha ", len(self.W_out[0]))
        #print("gout sh ", g_out.shape)
        multipl_in_sum = 0
        for i in range(0, self.dim_hid):
            multipl_in_sum += self.W_out[:,i] * g_out
            #self.W_out[i]
        result = multipl_in_sum * self.df_hid(a)
        return result
            

    def backward(self, x, d):       # x is input , d is real output
        y, b, h, a = self.forward(x)        # y is predicted output
        #print("a shape ", a.shape)
        g_out = (d - y) * self.f_hid(b) # FIXME
        #print ("gout shape ", h.shape)
        g_hid = self.compute_ghid(g_out, a) # rovnaky rozmer ako h
        #print("ghid sh ", g_hid.shape)
        #print("h sh ", h.shape)
        dW_out = np.dot(np.atleast_2d(g_out).T, np.atleast_2d(h)) # FIXME
        dW_hid = np.dot(np.atleast_2d(g_hid).T, np.atleast_2d(x)) # FIXME
        #dW_out = np.outer(g_out, h)
        #dW_hid = np.outer(g_hid, x)
        
        print("gout sh ", g_out.shape)
        print("ghid sh ", g_hid.shape)
        print("dwout sh ", dW_out.shape)
        print("dwhid sh ", dW_hid.shape)
        print("y sh ", y.shape)

        return y, dW_hid, dW_out

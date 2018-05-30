import numpy as np

from util import *


## Multi-Layer Perceptron
# (abstract base class)

class MLP():

    def __init__(self, dim_in, dim_hid, dim_out, validation_data, validation_label):
        self.dim_in     = dim_in
        self.dim_hid    = dim_hid
        self.dim_out    = dim_out
        self.dim_mid  	= 20
        self.validation_data = validation_data
        self.validation_label = validation_label

        self.W_hid = np.random.randn(dim_hid, dim_in + 1) # FIXME
        self.W_firstmid = np.random.randn(self.dim_mid, dim_hid + 1)		#vrstva bude medzi hidden a output, cize output hidden je input
        #self.W_out = np.random.randn(dim_out, dim_hid + 1) # old 
        self.W_out = np.random.randn(dim_out, self.dim_mid + 1)	# new with fisrt_mid layer
        #print ('whid shape', self.W_hid.shape)
        #print ('wout shape', self.W_out.shape)

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
        
    def stablesoftmax(self, x):
        pass
    def der_softmax(self, x):
        pass


    ## forward pass
    # (single input vector)

    def forward(self, x):
        # forward pass 
        a = np.dot(self.W_hid, augment(x)) #
        h = self.f_hid(a) #
        
        # W_firstmid
        f_mid = np.dot(self.W_firstmid, augment(h))
        output_f_mid = self.f_hid(f_mid)		# use same activation func as ofr hidden layer
        
        b = np.dot(self.W_out, augment(f_mid))
        #b = np.dot(self.W_out, augment(h)) # FIXME
        y = self.stablesoftmax(b) # FIXME

        return y, b, output_f_mid, f_mid, h, a


    ## forward & backprop pass
    # (single input and target vector)

    def backward(self, x, d):
        y, b, output_f_mid, f_mid, h, a = self.forward(x)

        g_out = (d - y) * self.df_out(b) # 
        
        g_mid = np.dot(self.W_out.T[:-1], g_out) * self.df_hid(a)
        
        g_hid = np.dot(self.W_firstmid.T[:-1], g_mid) * self.df_hid(f_mid) # 
        #g_hid = np.dot(self.W_out.T[:-1], g_out) * self.df_hid(a) # 

        #dW_out = np.outer(g_out, augment(h)) # out of output layer -> weights update in fact
        dW_out = np.outer(g_out, augment(output_f_mid))
        dW_mid = np.outer(g_mid, augment(h)) 
        dW_hid = np.outer(g_hid, augment(x)) 
        #dW_hid = np.outer(g_hid, augment(x)) # out of hidden layer -> weight update

        return y, dW_hid, dW_mid, dW_out

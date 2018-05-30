import numpy as np

from util import *


## Multi-Layer Perceptron
# (abstract base class)

class MLP():

    def __init__(self, dim_in, dim_hid, dim_out):
        self.dim_in     = dim_in
        self.dim_hid    = dim_hid
        self.dim_out    = dim_out

        self.W_hid = np.random.rand(dim_hid, dim_in + 1)
        self.W_out = np.random.rand(dim_out, dim_hid + 1)
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
        a = np.dot(self.W_hid, augment(x))
        h = self.f_hid(a)
        b = np.dot(self.W_out, augment(h))
        y = self.f_out(b)

        """
        print("shape W_hid: ", end="")
        print(np.shape(self.W_hid))
        print("shape a: ", end="")
        print(np.shape(a))
        print("shape h: ", end="")
        print(np.shape(h))
        print("shape W_out: ", end="")
        print(np.shape(self.W_out))
        print("shape b: ", end="")
        print(np.shape(b))
        print("shape y: ", end="")
        print(np.shape(y))
        """

        return y, b, h, a


    ## forward & backprop pass
    # (single input and target vector)

    def backward(self, x, d):
        print("x shape ", x.shape)
        y, b, h, a = self.forward(x)

        g_out = (d - y)*self.df_out(b)

        """
        print("\n")
        print("shape d: ", end = "")
        print(np.shape(d))
        print("shape g_out: ", end = "")
        print(np.shape(g_out))
        print("shape W_out: ", end = "")
        print(np.shape(self.W_out))
        print("shape df_hid(a): ", end = "")
        print(np.shape(self.df_hid(a)))
        """

        g_hid = np.dot(self.W_out.T[:-1], g_out) * self.df_hid(a)



        dW_out = np.outer(g_out, augment(h))
        dW_hid = np.outer(g_hid, augment(x))

        """
        print("shape g_hid: ", end = "")
        print(np.shape(g_hid))
        print("shape dW_hid: ", end = "")
        print(np.shape(dW_hid))
        print("shape dW_out: ", end = "")
        print(np.shape(dW_out))
        """

        return y, dW_hid, dW_out

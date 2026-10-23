import numpy as np
from .NN_Core import Tensor
from . import NN_Operations as operations


# Core "Module" Class
class Module:

    def __init__(self):
        self._modules = {}
        self._parameters = {}
    
    # overriding the setattr func to add some checking
    def __setattr__(self, name, value):
        if isinstance(value, Module):
            self._modules[name] = value
        elif isinstance(value, Tensor) and value.requires_grad:
            self._parameters[name] = value
        
        super().__setattr__(name, value)
    
    def __call__(self, *args):
        return self.forward(*args)




# Using "Module" Class - Layers

class Linear(Module):

    def __init__(self, n_in, n_out):
        super().__init__()

        # self.n_in = n_in
        # self.n_out = n_out/
        
        W = np.random.randn(n_out, n_in) * np.sqrt(2.0 / n_in) # shape = [output x input]
        b = np.zeros((n_out, 1))

        # print ("Linear Module W:", W)
        # print ("Linear Module b:", b)
        # print ("\n\n")

        self.weight = Tensor(W, requires_grad=True)
        self.bias = Tensor(b, requires_grad=True)
    
    def forward(self, x: Tensor):
        op = operations.LinearOp(x, self.weight, self.bias)
        return op.forward()


class ReLU(Module):
    
    def forward(self, x):
        op = operations.ReLUOp(x)
        return op.forward()


# class Softmax(Module):

#     def forward(self, x):
#         op = operations.SoftmaxOp(x)
#         return op.forward



# Types of Networks

class Sequential(Module):
    
    def __init__(self, *args): # *args = layers
        super().__init__()
        for (idx, module) in enumerate(args):
            self._modules[str(idx)] = module
    
    def forward(self, x):
        # counter = 1
        for module in self._modules.values():
            # print ("counter", counter)
            x = module(x)
            # counter += 1
        return x

    def parameters(self):
        parameters = {}
        # counter = 1
        for module_key in self._modules.keys():
            # parameters[f"Layer_{counter}"] = module._parameters
            parameters[f"Layer_{module_key}"] = self._modules[module_key]._parameters

        return parameters
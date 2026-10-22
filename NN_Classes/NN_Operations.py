import numpy as np
from .NN_Core import Tensor


class LinearOp:
    
    def __init__(self, x: Tensor, weight: Tensor, bias: Tensor):
        self.x = x
        self.weight = weight
        self.bias = bias
    
    def forward(self):
        # out_data = (self.x.data @ self.weight.data) + self.bias.data
        out_data = (self.weight.data @ self.x.data) + self.bias.data
        # print ("linear, forward, out_data.shape", out_data.shape)
        return Tensor(out_data, creator=self)
    
    def backward(self, grad):

        # print ("grad[0]", grad[0])
        
        if self.weight.requires_grad:
            # print ("updating weight\n", "\tgrad.shape", grad.shape, "\n\t", "self.x.data.T.shape", self.x.data.T.shape)
            # print ("updating weight\n", "\tself.weight.grad.shape", self.weight.grad.shape)
            self.weight.grad += grad @ self.x.data.T
        
        if self.bias.requires_grad:
            # print ("updating bias\n", "\tself.bias.grad.shape", self.bias.grad.shape, "\tgrad.shape", grad.shape)
            # print ("updating bias\n", "\tnp.sum(grad, axis=1).shape", np.sum(grad, axis=1, keepdims=True).shape)
            self.bias.grad += np.sum(grad, axis=1, keepdims=True)
            # print ("BIAS DONE")

        # print ("self.x.requires_grad", self.x.requires_grad)
        
        if self.x.requires_grad:
            # print ("linear, backward\n", "\tself.weight.data.T.shape", self.weight.data.T.shape, "\n\tgrad.shape", grad.shape)
            # print ("linear, backward, weight @ grad.shape", (self.weight.data.T @ grad).shape)
            self.x.backward(self.weight.data.T @ grad)


class ReLUOp:
    
    def __init__(self, x: Tensor):
        self.x = x

    
    def forward(self):
        out_data = np.maximum(0.0, self.x.data)
        # print ("relu, forward, out_data.shape", out_data.shape)
        return Tensor(out_data, creator=self)
    
    def backward(self, grad):
        if self.x.requires_grad:
            relu_derivative = (self.x.data > 0).astype(np.float32)
            # print ("relu, backward, grad * relu_derivative.shape", (grad * relu_derivative).shape)
            self.x.backward(grad * relu_derivative) # Hadamard Product


# class SoftmaxOp:

#     def __init__(self, Z: Tensor):
#         self.Z = Z

#     def forward(self):
#         max_val = np.max(self.Z, axis=0, keepdims=True)
#         exponents = np.exp(self.Z - max_val) # to keep stable and prevent overflow
#         softmax_result = exponents / np.sum(exponents, axis=0, keepdims=True)
#         return Tensor(softmax_result, creator=self, requires_grad=False)

#     def backward(self, grad):
#         # doesn't require gradient
#         if self.Z.requires_grad:
#             self.Z.backward(grad)



# class CategoricalCrossEntropyLossOp:

#     def __init__(self, predictions, target):
#         self.pred = predictions
#         self.target = target

#     def forward(self): # calculate loss
#         error = self.predictions - self.target
#         return Tensor(error, creator=self)

#     def backward(self, grad):
#         pass


class SoftmaxCrossEntropyLossOp:

    def __init__(self, raw_logits: Tensor, targets: Tensor):
        self.raw_logits = raw_logits
        self.targets = targets
        self.probabilities = None

    # just calculate loss
    def forward(self):
        shifted_raw_logits = self.raw_logits.data - np.max(self.raw_logits.data, axis=0, keepdims=True) # to keep stable and prevent overflow
        exponents = np.exp(shifted_raw_logits)
        self.probabilities = exponents / np.sum(exponents, axis=0, keepdims=True)

        eps = 1e-15
        self.probabilities = np.clip(self.probabilities, eps, 1 - eps) # for numerical stability (ensures no log(0)) AND np.clip just bounds everything
        # loss = np.mean(np.sum(self.targets.data * np.log(self.probabilities.T)))
        loss = np.mean(-np.sum(self.targets.data * np.log(self.probabilities), axis=0))

        # print ("loss.shape", loss.shape)


        return Tensor(loss, creator=self)

    def backward(self, grad):

        if self.raw_logits.requires_grad:
            assert self.probabilities.data.shape == self.targets.data.shape
            batch_size = self.raw_logits.data.shape[1] # CHECK
            # print ("batch_size", batch_size)
            error_gradient = (self.probabilities - self.targets.data) / batch_size
            # print ("backward: first error gradient.shape", error_gradient.shape)
            # self.raw_logits.backward(grad * error_gradient)
            # print ("backward: first error gradient * grad.shape", (error_gradient * grad).shape)
            self.raw_logits.backward(error_gradient * grad)
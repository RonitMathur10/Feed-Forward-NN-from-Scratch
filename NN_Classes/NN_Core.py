import numpy as np

class Tensor:
    
    def __init__(self, data, creator=None, requires_grad=True):
        self.data = data
        self.creator = creator
        self.requires_grad = requires_grad
        # self.grad = 0.0 if requires_grad else None
        self.grad = np.zeros_like(self.data) if requires_grad else None
    
    def backward(self, grad=None):
        
        if not self.requires_grad:
            return
        
        if grad is None:
            grad = np.ones_like(self.data)
            # print ("grad.shape (first gradient - np.ones_like)", grad.shape)
        
        self.grad += grad

        if self.creator is not None:
            self.creator.backward(grad)
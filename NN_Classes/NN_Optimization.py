import numpy as np


class SGD:

    def __init__(self, model, learning_rate=0.01):
        self.model = model
        self.lr = learning_rate

    # recursive
    def _get_all_parameters(self, curr_module): # returns all the tracked parameters (requires_grad) in the model - will be used for updates
        params = list(curr_module._parameters.values()) # type = list[Tensors] (i think that's it)
        for sub_module in curr_module._modules.values():
            params.extend(self._get_all_parameters(sub_module))
        return params

    # reset params
    def zero_grad(self):
        params = self._get_all_parameters(self.model)
        for param in params:
            param.grad.fill(0.0) # just rewrite all gradients back to 0

    # update params
    def step(self):
        # counter = 1
        params = self._get_all_parameters(self.model)
        for param in params:
            # print ("step, counter:", counter)
            # counter += 1
            param.data -= self.lr * param.grad
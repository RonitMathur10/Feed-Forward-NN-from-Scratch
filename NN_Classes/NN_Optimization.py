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



class Adam:

    def __init__(self, model, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-7):
        self.model = model
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.t = 0 # global time-step counter

        # dictionaries to keep track of past gradients
        self.m = {} # first moment
        self.v = {} # second moment

    # reset params
    def zero_grad(self):
        # params = self._get_all_parameters(self.model)
        params = self.model.parameters()
        for param in params.values():
            param.grad.fill(0.0) # just rewrite all gradients back to 0

    # update params
    def step(self):

        self.t += 1

        model_parameters = self.model.parameters()

        for (name, param) in model_parameters.items():

            if param.grad is None: # the parameter gradient isn't being tracked
                continue

            curr_gradient = param.grad

            # initialize with 0s if the layer hasn't been computed yet
            if name not in self.m:
                self.m[name] = np.zeros_like(param.data)
                self.v[name] = np.zeros_like(param.data)

            self.m[name] = (self.beta1 * self.m[name]) + ((1 - self.beta1) * curr_gradient)
            self.v[name] = (self.beta2 * self.v[name]) + ((1 - self.beta2) * (np.square(curr_gradient)))

            m_hat = self.m[name] / (1.0 - (self.beta1 ** self.t))
            v_hat = self.v[name] / (1.0 - (self.beta2 ** self.t))

            param.data -= self.learning_rate * (m_hat / (np.sqrt(v_hat) + self.epsilon))
import numpy as np


class NN():

    def __init__(self, X_train: np.ndarray, Y_train: np.ndarray, X_test: np.ndarray, Y_test: np.ndarray, hidden_layer_sizes: np.ndarray):
        # data should be: rows=features, cols=datapoints
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.one_hot_Y = NN.one_hot(self.Y_train) # shape: [n x c] --> rows=each datapoint , cols=each class possibility (ie. 0, 1, 2)
        
        self.num_train_data_points = X_train.shape[1]
        self.num_test_data_points = X_test.shape[1]
        self.num_data_points = self.num_train_data_points + self.num_test_data_points
        self.num_hidden_layers = len(hidden_layer_sizes) # one number to indicate # of hidden layers (excluding input & output layers)
        self.total_num_layers = self.num_hidden_layers + 2
        self.layer_sizes = np.append(np.append(X_train.shape[0], hidden_layer_sizes), len(np.unique(Y_train))) # array containing every layer size (including input & output layer)
        
        # self.rng = np.default_rng(seed=42)
        np.random.seed(42)

        self.best_model_params = {"epoch": -1, "accuracy": -1, "weights": None, "biases": None}

        self.init_network()

    def init_network(self):
        print ("Initializing Network")
        self.init_learnable_params()
        self.init_model_functions()
        print ("Network Initialized")
    
    def init_learnable_params(self):
        self.weights = [] # len = L - 1 (L = # all layers)
        self.biases = [] # len = L - 1 (L = # all layers)
        print ("Initializing Learnable Parameters")
        for index in range(1, self.total_num_layers):
            n_out = int(self.layer_sizes[index])
            n_in = int(self.layer_sizes[index-1])
            # W = np.random.randn(n_out, n_in) # shape = [output x input]
            W = np.random.randn(n_out, n_in) * np.sqrt(2.0 / n_in) # shape = [output x input]
            b = np.zeros((n_out, 1))
            # b = np.full(n_out, 1.0).reshape(n_out, 1)
            
            self.weights.append(W)
            self.biases.append(b)
        print ("Learnable Parameters Initialized")
    
    def init_model_functions(self):
        print ("Initializing Model Functions")
        self.activation_function = NN.ReLU
        self.activation_function_derivative = NN.ReLU_derivative
        # ADD OPTIMIZER, ETC.
        print ("Model Functions Initialized - change as needed")


    def train(self, epochs, learning_rate):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.accuracies_over_all_epochs = np.array([])
        self.loss_over_all_epochs = np.array([])
        self.best_accuracy = -1
        
        for epoch in range(epochs):
            
            self.forward_prop()
            self.back_prop()

            prediction_accuracy = self._predict_for_training()
            loss = self.calc_loss(self.activations[-1])
            self.accuracies_over_all_epochs = np.append(self.accuracies_over_all_epochs, prediction_accuracy)
            self.loss_over_all_epochs = np.append(self.loss_over_all_epochs, loss)
    
            if epoch % 10 == 0:
                print ("---------------------------------")
                print (f"Epoch: {epoch}")
                print (f"Accuracy: {prediction_accuracy}")
                print (f"Loss: {loss}")

            if prediction_accuracy > self.best_model_params["accuracy"]:
                self.best_model_params = {"epoch": epoch, "accuracy": prediction_accuracy, "weights": self.weights.copy(), "biases": self.biases.copy()}
        
        self.final_model_params = {"epoch": epochs, "accuracy": prediction_accuracy, "weights": self.weights.copy(), "biases": self.biases.copy()}

        # print ("\n\n", f"epoch # and accuracy of highest accuracy epoch: {np.argmax(self.accuracies_over_all_epochs)}, {self.accuracies_over_all_epochs[np.argmax(self.accuracies_over_all_epochs)]}")
        # "BEST" MODEL DOES NOT CONSIDER LOSS, ONLY ACCURACY
        print ("\n\n", f"Highest Accuracy:\n\tEpoch # = {self.best_model_params["epoch"]}\n\tAccuracy = {self.best_model_params["accuracy"]}\n\tLoss = {self.loss_over_all_epochs[self.best_model_params["epoch"]]}")
        



    # Core Functions
    
    def forward_prop(self):

        # self.weighted_sum_matrices = [self.X_train] # all of the Z matrices (used for backprop)
        self.weighted_sum_matrices = []
        self.activations = [self.X_train] # training_set + activations
        
        # for (index, weight_matrix) in enumerate(self.weights):
        for index in range(self.num_hidden_layers):
            Z = self.weights[index] @ self.activations[index] + self.biases[index] # Z = linear sum --> (W @ X) + b
            self.weighted_sum_matrices.append(Z)
            # A = NN.ReLU(Z) # A = activation of Z
            A = self.activation_function(Z) # A = activation of Z 
            self.activations.append(A)

        # for the output layer
        Z = self.weights[-1] @ self.activations[-1] + self.biases[-1]
        self.weighted_sum_matrices.append(Z)
        
        output_probability_matrix_wrong_shape = NN.softmax(Z, axis=0) # softmax activation instead of ReLU, shape: rows=features , cols=datapoints
        self.activations.append(output_probability_matrix_wrong_shape)

        for i, Z in enumerate(self.weighted_sum_matrices[:-1]):
            dead_frac = np.mean(Z <= 0)
            print(f"layer {i}: {dead_frac:.2%} dead neurons")


        # each matrix in self.forward_prop_inputs has shape: [output_neurons x datapoints]

    def back_prop(self):
        # one_hot_Y = NN.one_hot(self.Y_train) # shape: [n x c] --> rows=each datapoint , cols=each class possibility (ie. 0, 1, 2)
        self.output_probability_matrix = self.activations[-1].T # shape: rows=datapoints , cols=features

        self.weight_errors = [None] * (self.total_num_layers - 1)
        self.bias_errors = [None] * (self.total_num_layers - 1)

        assert self.output_probability_matrix.shape == self.one_hot_Y.shape
        error_gradient = self.output_probability_matrix - self.one_hot_Y # shape = [n x c]
        error_gradient = error_gradient.T # shape = [c x n]

        dW = (np.dot(error_gradient, self.activations[-2].T)) / self.num_train_data_points
        dB = np.sum(error_gradient, axis=1, keepdims=True) / self.num_train_data_points
        self.weight_errors[-1] = dW
        self.bias_errors[-1] = dB
        
        # for index in range(self.total_num_layers - 2, -1, -1):
        for index in reversed(range(self.num_hidden_layers)):

            prev_layer_weighted_sum_matrix = self.weighted_sum_matrices[index]
            prev_layer_activation_matrix = self.activations[index].T

            # error_gradient = np.dot(self.weights[index+1].T, error_gradient) * NN.ReLU_derivative(prev_layer_weighted_sum_matrix)
            error_gradient = np.dot(self.weights[index+1].T, error_gradient) * self.activation_function_derivative(prev_layer_weighted_sum_matrix)

            dW = np.dot(error_gradient, prev_layer_activation_matrix) / self.num_train_data_points
            dB = np.sum(error_gradient, axis=1, keepdims=True) / self.num_train_data_points

            self.weight_errors[index] = dW
            self.bias_errors[index] = dB

        self.update_parameters()

    def update_parameters(self):
        for index in range(self.total_num_layers - 1):
            self.weights[index] -= self.learning_rate * self.weight_errors[index]
            self.biases[index] -= self.learning_rate * self.bias_errors[index]


    def _predict_for_training(self):
        
        forward_prop_prediction_inputs = [self.X_test] # training_set + activations
        
        # forward propagation process
        for index in range(self.num_hidden_layers):
            Z = self.weights[index] @ forward_prop_prediction_inputs[index] + self.biases[index] # Z = linear sum --> (W @ X) + b
            # A = NN.ReLU(Z) # A = activation of Z
            A = self.activation_function(Z)
            forward_prop_prediction_inputs.append(A)
        # for the output layer
        Z = self.weights[-1] @ forward_prop_prediction_inputs[-1] + self.biases[-1]
        prediction_output_probability_matrix = NN.softmax(Z.T, axis=1)

        prediction_output_classes = np.argmax(prediction_output_probability_matrix + np.min(self.Y_train), axis=1)

        correct_predictions = prediction_output_classes == self.Y_test.squeeze() # using squeeze to ensure dimensions match
        # final_prediction_accuracy = sum(correct_predictions) / self.num_test_data_points
        final_prediction_accuracy = np.mean(correct_predictions) # np.mean instead of sum() faster for larger datasets

        # self.accuracies_over_all_epochs = np.append(self.accuracies_over_all_epochs, final_prediction_accuracy)

        return final_prediction_accuracy
        
        
        

    # Helper Functions
    
    def ReLU(Z):
        return np.maximum(0, Z)

    def ReLU_derivative(Z):
         return (Z > 0).astype(float)

    def leaky_ReLU(Z, alpha=0.01):
        return np.maximum(alpha*Z, Z)

    def leaky_ReLU_derivative(Z, alpha=0.01):
         # return (Z > 0).astype(float)
        return np.where(Z > 0, 1, alpha)

    def softmax(Z, axis):
        # Z = Z - np.max(matrix, axis=1, keepdims=True)
        # return np.exp(Z) / np.sum(np.exp(Z), axis=1, keepdims=True)
        max_val = np.max(Z, axis=axis, keepdims=True)
        exponents = np.exp(Z - max_val) # to keep stable and prevent overflow
        return exponents / np.sum(exponents, axis=axis, keepdims=True)
    
    def one_hot(Y):
        Y = Y.astype(int)
        # one_hot_Y = np.zeros(( Y.size, (np.max(Y) - np.min(Y) + 1) ))   # = array with dimensions: [ (n) , (# of classes, which is 10) ]
        one_hot_Y = np.zeros(( Y.size, len(np.unique(Y)) ))   # = array with dimensions: [ (n) , (# of classes, which is 10) ]
        one_hot_Y[np.arange(Y.size), Y - np.min(Y)] = 1
        return one_hot_Y
    
    def calc_loss(self, outputs):
        eps = 1e-15
        outputs = np.clip(outputs, eps, 1 - eps) # for numerical stability (ensures no log(0)) AND np.clip just bounds everything
        return -(1/outputs.shape[1]) * np.sum(self.one_hot_Y * np.log(outputs.T))
    


    # Setter Functions
    def set_weights_and_biases(self, type: str):
        if type == "best":
            self.weights = self.best_model_params["weights"]
            self.biases = self.best_model_params["biases"]
        elif type == "final":
            self.weights = self.final_model_params["weights"]
            self.biases = self.final_model_params["biases"]
    
    def set_activation(self, function: str):
        if function == "ReLU":
            self.activation_function = NN.ReLU
            self.activation_function_derivative = NN.ReLU_derivative
        elif function == "leaky_ReLU":
            self.activation_function = NN.leaky_ReLUReLU
            self.activation_function_derivative = NN.leaky_ReLU_derivative



    # Final Predict Function!
    def predict(self, X_data=None, Y_data=None):

        if X_data is None:
            forward_prop_prediction_inputs = [self.X_test] # training_set + activations
        else:
            forward_prop_prediction_inputs = [X_data] # training_set + activations
        
        # forward propagation process
        for index in range(self.num_hidden_layers):
            Z = self.weights[index] @ forward_prop_prediction_inputs[index] + self.biases[index] # Z = linear sum --> (W @ X) + b
            # A = NN.ReLU(Z) # A = activation of Z
            A = self.activation_function(Z)
            forward_prop_prediction_inputs.append(A)
        # for the output layer
        Z = self.weights[-1] @ forward_prop_prediction_inputs[-1] + self.biases[-1]
        prediction_output_probability_matrix = NN.softmax(Z.T, axis=1)

        # prediction_output_classes = np.argmax(prediction_output_probability_matrix + np.min(self.Y_train), axis=1)
        prediction_output_classes = np.argmax(prediction_output_probability_matrix, axis=1) # WHAT TO DO FOR OFFSET?

        if Y_data is not None:
            correct_predictions = prediction_output_classes == Y_data.squeeze() # using squeeze to ensure dimensions match
            # final_prediction_accuracy = sum(correct_predictions) / self.num_test_data_points
            final_prediction_accuracy = np.mean(correct_predictions) # np.mean instead of sum() faster for larger datasets
            return final_prediction_accuracy
        else:
            return prediction_output_classes
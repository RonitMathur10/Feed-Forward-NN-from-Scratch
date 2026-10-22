import numpy as np
from Feed_Forward_NN_Class import NN

class load_NN(NN):
    def __init__(self, model_name: str, data_location: str):
        if data_location == "Kaggle_Digits":
            data_path = "/Users/ronitmathur/Desktop/ML/Feed-Forward_NN/Data/Processed_Data/Kaggle_Digits"

            X_train = np.load(f"{data_path}/X_train.npy")
            Y_train = np.load(f"{data_path}/Y_train.npy")
            X_validation = np.load(f"{data_path}/X_validation.npy")
            Y_validation = np.load(f"{data_path}/Y_validation.npy")
            self.X_test = np.load(f"{data_path}/X_test.npy").T
        
        saved_model_path = f"/Users/ronitmathur/Desktop/ML/Feed-Forward_NN/Kaggle/Results/{model_name}"
        layer_sizes = np.load(f"{saved_model_path}/layer_sizes.npy")


        data = np.load(f"{saved_model_path}/weights_and_biases.npz")
        number_layers_for_wb = np.size(layer_sizes) - 1
        
        # When passed as positional arguments (*), arrays are named 'arr_0', 'arr_1', etc.
        # Total positional items = w_count + b_count
        all_arrays = [data[f'arr_{i}'] for i in range(2*number_layers_for_wb)]
        
        # Slice the combined list back into weights and biases
        weights = all_arrays[:number_layers_for_wb]
        biases = all_arrays[number_layers_for_wb:]
        
        
        super().__init__(model_name, X_train.T, Y_train.T, X_validation.T, Y_validation.T, layer_sizes[1:-1])
        self.set_loaded_weigths_and_biases(weights, biases)


        print ("Model Loaded")

        # DO:
        #   2. create setter function in class.py to set weights & biases
        #   3. create system to save trained models into respective folders
        # ---- EVERYTHING DONE I THINK, JUST CHECK WITH TESTING. also replace code in notebook to just load the kaggle data instead of calc. everything again ----
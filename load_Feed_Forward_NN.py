import numpy as np
from Feed_Forward_NN_Class import NN

class load_NN(NN):
    def __init__(self, model_name: str, data_location: str):
        if data_location == "Kaggle_Digits":
            data_path = "/Users/ronitmathur/Desktop/ML/Feed-Forward_NN/Data/Processed_Data/Kaggle_Digits"

            X_train = np.load(f"{data_path}/X_train")
            Y_train = np.load(f"{data_path}/Y_train")
            X_validation = np.load(f"{data_path}/X_validation")
            Y_validation = np.load(f"{data_path}/Y_validation")
            self.X_test = np.load(f"{data_path}/X_test")
        
        super.__init__(model_name, X_train, Y_train, X_validation, Y_validation)

        # DO:
        #   1. add code to save 1. layer sizes , 2. model's weights & biases
        #   2. create setter function in class.py to set weights & biases
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class Plot():

    def __init__(self, model):
        self.model = model
        self.epochs = list(range(self.model.epochs))
    

    def plot_all(self):
        self.plot_accuracy()
        self.plot_loss()
        self.plot_dead_neurons()   

    def plot_accuracy(self):
        plt.figure()
        sns.lineplot(x=self.epochs, y=self.model.accuracies_over_all_epochs)
        plt.title("Accuracy vs. Epoch")
        plt.show()
    
    def plot_loss(self):
        plt.figure()
        sns.lineplot(x=self.epochs, y=self.model.loss_over_all_epochs)
        plt.title("Loss vs. Epoch")
        plt.show()
    
    def plot_dead_neurons(self):
        plt.figure()
        dead_neurons_combined = np.array([self.model.dead_neurons_per_layer[i] for i in self.model.dead_neurons_per_layer.keys()])
        dead_neurons_combined_df = pd.DataFrame(dead_neurons_combined.T, columns=['layer_1', 'layer_2', 'layer_3', 'layer_4'])
        dead_neurons_combined_df["epochs"] = self.epochs
        dead_neurons_combined_df.set_index("epochs")
        sns.lineplot(data=dead_neurons_combined_df.iloc[:, :-1])
        plt.title("Dead Neurons per Layer (%) vs. Epoch")
        plt.show()
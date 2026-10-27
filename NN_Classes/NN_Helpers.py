import numpy as np
import pandas as pd


def one_hot(Y):
    Y = Y.astype(int) # one_hot_Y = np.zeros(( Y.size, (np.max(Y) - np.min(Y) + 1) ))=      # = array with dimensions: [ (n) , (# of classes, which is 10) ]
    one_hot_Y = np.zeros(( Y.size, len(np.unique(Y)) ))   # = array with dimensions: [ (n) , (# of classes, which is 10) ]
    one_hot_Y[np.arange(Y.size), Y - np.min(Y)] = 1
    return one_hot_Y


def get_model_wandb(model):
    model_parameters = model.parameters()
    weights = []
    biases = []
    # for param in model_parameters.keys():
    #     temp = model_parameters[param]
    #     if len(temp) > 0:
    #         weights.append(temp["weight"].data)
    #         biases.append(temp["bias"])
    for param in model_parameters.keys():
        if param[-1] == "t":
            weights.append(model_parameters[param].data)
        elif param[-1] == "s":
            biases.append(model_parameters[param].data)
    return (weights, biases)


def calc_dead_neurons(weights):
    for i, Z in enumerate(weights):
        dead_frac = np.mean(Z <= 0)
        print(f"layer {i}: {dead_frac:.2%} dead neurons")


def predict(model, data):
    # x_test_data = nn.Tensor(model.T, requires_grad=False)
    output_logits = model(data)
    shifted_output_logits = output_logits.data - np.max(output_logits.data, axis=0, keepdims=True) # to keep stable and prevent overflow
    exponents = np.exp(shifted_output_logits)
    probabilities = (exponents / np.sum(exponents, axis=0, keepdims=True)).T

    predictions = np.argmax(probabilities, axis=1)
    return predictions


def predict_save_kaggle(model, data, folder_path):
    predictions = predict(model, data)
    # flatten and ensure predictions are integers (0 instead of 0.0)
    final_predictions = np.asarray(predictions).flatten().astype(int)

    # create df and shift the index to start at 1 (kaggle required format)
    df = pd.DataFrame({'Label': final_predictions})
    df.index = df.index + 1

    # save to a CSV
    df.to_csv(f"/Users/ronitmathur/Desktop/ML/Feed-Forward_NN/Kaggle/Results/{folder_path}.csv", index_label='ImageId')
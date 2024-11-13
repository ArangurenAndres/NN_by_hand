# Assignment 1 Question 4
import numpy as np
from urllib import request
import gzip
import pickle
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns

def load_config(default = True):
    if default:
        config_path = os.path.join(os.getcwd(),"config.json")
        with open(config_path, "r") as jsonfile:
            data = json.load(jsonfile)
        print("Read successful")
    else:
        print("Processing custom config file")
    return data

def load_synth(num_train=60_000, num_val=10_000, seed=0):
    """
    Load some very basic synthetic data that should be easy to classify. Two features, so that we can plot the
    decision boundary (which is an ellipse in the feature space).

    :param num_train: Number of training instances
    :param num_val: Number of test/validation instances
    :param num_features: Number of features per instance

    :return: Two tuples and an integer: (xtrain, ytrain), (xval, yval), num_cls. The first contains a matrix of training
     data with 2 features as a numpy floating point array, and the corresponding classification labels as a numpy
     integer array. The second contains the test/validation data in the same format. The last integer contains the
     number of classes (this is always 2 for this function).
    """
    np.random.seed(seed)

    THRESHOLD = 0.6
    quad = np.asarray([[1, -0.05], [1, .4]])

    ntotal = num_train + num_val

    x = np.random.randn(ntotal, 2)

    # compute the quadratic form
    q = np.einsum('bf, fk, bk -> b', x, quad, x)
    y = (q > THRESHOLD).astype(int)

    return (x[:num_train, :], y[:num_train]), (x[num_train:, :], y[num_train:]), 2

def norm_data(x):
    min_value = x.min()
    max_value = x.max()
    norm_x = (x-min_value)/(max_value-min_value)
    return norm_x


def moving_average(data,window_size=5):
    #apply a convolution to 
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def plot_loss(train_loss,val_loss, visualize=True, save = False, filename = None, window_size=5):
    x = [x for x in range(len(train_loss))]
    # Apply moving average to smooth the loss curve
    smoothed_loss = moving_average(train_loss, window_size=window_size)
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=x, y=train_loss, color="royalblue", linewidth=2.5, markersize=6, label=f"Training loss")
    sns.lineplot(x=x, y=val_loss, color="red", linewidth=2.5, markersize=6, label=f"Test loss")

    #sns.lineplot(x=x[window_size-1:], y=smoothed_loss, color="orange", linewidth=2.5, label=f"Smoothed Loss (window={window_size})")

    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Loss vs. Epochs", fontsize=14, fontweight='bold')
    plt.grid(visible=True, linestyle="--", alpha=0.7)
    # Add a legend
    plt.legend()    
        
    if save:
        plt.savefig(os.path.join("results/",filename))
    if visualize:
        plt.show()
    else:
        pass


if __name__ == "__main__":
    train,val,_ = load_synth(num_train=60_000, num_val=10_000, seed=0)
    x_train,y_train = train
    x_val,y_val = val
    # Normalize data
    x_train = norm_data(x_train)
    x_val = norm_data(x_val)
    print(x_train[:5])


    #for x_,y_ in zip(x_train[:5],y_train[:5]):
    #    print(x_,y_)



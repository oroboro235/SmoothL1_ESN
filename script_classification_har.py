import time
import os
import sys
sys.path.append(os.path.join(os.getcwd(), 'L1General_python'))
import random
random.seed(1234)

import numpy as np
import pandas as pd
np.random.seed(1234)
import matplotlib.pyplot as plt

# esn model
from models.esn_model import esn_classification

from functools import partial
import multiprocessing
from multiprocessing import Pool

def preprocess_uci_har(data_dir):
    """Preprocess the UCI HAR dataset for sequence prediction task"""
    
    # list of sensor signal file names
    sensor_signals = [
        'body_acc_x', 'body_acc_y', 'body_acc_z',
        'body_gyro_x', 'body_gyro_y', 'body_gyro_z',
        'total_acc_x', 'total_acc_y', 'total_acc_z'
    ]

    def load_dataset(subset):
        """loades the dataset of the given subset (train or test)"""
        features = []
        # load 9 sensor signals and stack them
        for signal in sensor_signals:
            file_path = os.path.join(
                data_dir, subset, 'Inertial Signals', 
                f'{signal}_{subset}.txt'
            )
            # load data and add channel dimension (samples, timesteps, 1)
            data = np.loadtxt(file_path)[:, :, np.newaxis]
            features.append(data)
        
        # concatenate all sensor features (samples, 128, 9)
        return np.concatenate(features, axis=-1)

    # load training and test sets
    X_train = load_dataset('train')
    X_test = load_dataset('test')

    # standardize: use statistics of training set
    mean = X_train.mean(axis=(0,1))  # calculate mean along sample and timesteps
    std = X_train.std(axis=(0,1))    # calculate std along sample and timesteps
    X_train = (X_train - mean) / (std + 1e-8) 
    X_test = (X_test - mean) / (std + 1e-8)

    # process labels (original labels are 1-6)
    def load_labels(subset):
        label_path = os.path.join(data_dir, subset, f'y_{subset}.txt')
        labels = np.loadtxt(label_path).astype(int) - 1  # convert to 0-5
        return labels
    # return np.eye(6)[labels]  # convert to one-hot encoding

    y_train = load_labels('train')
    y_test = load_labels('test')


    return X_train, y_train, X_test, y_test

def randomSearch(X, y, steps=100, size_r=100, sparsity=0.95, sr=None, lr=None, scale_i=None, params={}):
    # range of search
    param_range = {
        "sr" : list(np.arange(0.5, 5.0, 0.01)) if sr is None else [sr],
        "lr" : list(np.arange(0.1, 1.0, 0.01)) if lr is None else [lr],
        "scale_i" : (1e-2, 1e-1, 1e0, 1e1, 1e2) if scale_i is None else [scale_i],
	}
    max_acc = 0
    best_params = {}
    for _ in range(steps):
        sr = random.choice(param_range["sr"])
        lr = random.choice(param_range["lr"])
        scale_i = random.choice(param_range["scale_i"])
        esn = esn_classification(
            size_reservoir=size_r,
            spectral_radius=sr,
            sparsity=sparsity,
            leaking_rate=lr,
            input_scaling=scale_i,
            verbose=0,
            **params
        )
        acc = esn.fit(X, y)
        if acc > max_acc:
            max_acc = acc
            best_params = {
                "sr": sr,
                "lr": lr,
                "scale_i": scale_i,
            }
            print("Best params: {}".format(best_params))
    return best_params

if __name__ == '__main__':
    # load har
    datapath="./datasets/har"
    X_train, y_train, X_test, y_test = preprocess_uci_har(datapath)

    num_train_samples = 1000
    num_valid_samples = 200
    num_test_samples =  500


    # get index of random samples
    train_idx = np.random.permutation(X_train.shape[0])[:num_train_samples]
    # train_idx = np.random.permutation(X_train.shape[0])
    test_idx = np.random.permutation(X_test.shape[0])[:num_test_samples]
    # test_idx = np.random.permutation(X_test.shape[0])


    # select subset of data
    X_train = X_train[train_idx]
    y_train = y_train[train_idx]
    X_test = X_test[test_idx]
    y_test = y_test[test_idx]

    # split validation set
    X_valid = X_train[:num_valid_samples]
    y_valid = y_train[:num_valid_samples]
    X_train = X_train[num_valid_samples:]
    y_train = y_train[num_valid_samples:]

    params_search = {
        "size_r": 500,
        "sparsity": 0.95,
        "sr": None,
        "lr": None,
        "scale_i": None,
    }

    params_fixed = {
        "reg_type": "l2",
        "reg_param": 1e1,
        "num_warmup": 0,
        "evaluation_metric": "acc" 
    }

    params = {**params_search, **params_fixed}

    best_params = randomSearch(
        X_valid, 
        y_valid,
        steps=1000,
        size_r=100,
        sparsity=params_search["sparsity"],
        sr=params_search["sr"],
        lr=params_search["lr"],
        scale_i=params_search["scale_i"],
        params=params_fixed
    )

    params["sr"] = best_params["sr"]
    params["lr"] = best_params["lr"]
    params["scale_i"] = best_params["scale_i"]

    # params["sr"] = 0.8
    # params["lr"] = 0.5
    # params["scale_i"] = 1.0

    print(params)

    esn = esn_classification(
        size_reservoir=params["size_r"],
        spectral_radius=params["sr"],
        sparsity=params["sparsity"],
        leaking_rate=params["lr"],
        input_scaling=params["scale_i"],
        reg_type=params["reg_type"],
        reg_param=params["reg_param"],
        num_warmup=params["num_warmup"],
        evaluation_metric=params["evaluation_metric"]
    )

    acc = esn.fit(X_train, y_train)
    print("ACC in training set: {}".format(acc))

    y_pred = esn.predict(X_test)
    acc = esn.evaluate(y_pred, y_test)
    print("ACC in test set: {}".format(acc))
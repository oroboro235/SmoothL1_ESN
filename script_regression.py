import os
import sys
sys.path.append(os.path.join(os.getcwd(), 'L1General_python'))
import random

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

# esn model
from models.esn_model import esn_regression

from functools import partial
import multiprocessing
from multiprocessing import Pool


def randomSearch(X, y, len_valid, k=1, steps=100, size_r=100, sparsity=0.95, sr=None, lr=None, scale_i=None, reg_param=None, params={}):
    # range of search
    param_range = {
        "sr" : list(np.arange(0.05, 3.0, 0.05)) if sr is None else [sr],
        "lr" : list(np.arange(0.0, 1.0, 0.01)) if lr is None else [lr],
        "scale_i" : [1e-3, 1e-2, 1e-1, 1e0, 1e1] if scale_i is None else [scale_i],
        "reg_param": [1e-2, 1e-1, 1e0, 1e1, 1e2] if reg_param is None else [reg_param],
	}
    min_e = np.inf
    best_params = {}

    valid_indices = np.random.choice(len(X)-len_valid-1, k, replace=False).tolist()

    # generate random search params not replaced
    sampled_params = []
    i = 0
    while i < steps:
        sr = random.choice(param_range["sr"])
        lr = random.choice(param_range["lr"])
        scale_i = random.choice(param_range["scale_i"])
        reg_param = random.choice(param_range["reg_param"])
        if (sr, lr, scale_i, reg_param) not in sampled_params:
            sampled_params.append((sr, lr, scale_i, reg_param))
            i += 1
        else:
            continue
    
    for j, (sr, lr, scale_i, reg_param) in enumerate(sampled_params):
        print("Step {}/{}".format(j+1, steps))
        e_mean = 0.0
        for idx in valid_indices:
            X_valid = X[idx:idx+len_valid]
            y_valid = y[idx:idx+len_valid]
            split_idx = int(len(X_valid) * 0.8)
            X_tr, y_tr = X_valid[:split_idx], y_valid[:split_idx]
            X_ts, y_ts = X_valid[split_idx:], y_valid[split_idx:]
            esn = esn_regression(
                size_reservoir=size_r,
                spectral_radius=sr,
                sparsity=sparsity,
                leaking_rate=lr,
                input_scaling=scale_i,
                reg_param=reg_param,
                verbose=0,
                **params
            )
            e_tr = esn.fit(X_tr, y_tr)
            # evaluate the auto-regression model on test set
            e_ts = esn.evaluate(esn.generate(len(X_ts)), y_ts)
            e_mean += e_ts
        e_mean /= k
        if e_mean < min_e:
            min_e = e_mean
            best_params = {
                "sr": sr,
                "lr": lr,
                "scale_i": scale_i,
                "reg_param": reg_param
            }
            print("Best params: {}, error: {}".format(best_params, min_e))
    return best_params, min_e

def randomSearch_task(args, X, y, len_valid, valid_indices, size_r=100, sparsity=0.95, params={}):
    e_mean = 0.0
    for idx in valid_indices:
        X_valid = X[idx:idx+len_valid]
        y_valid = y[idx:idx+len_valid]
        split_idx = int(len(X) * 0.8)
        X_tr, y_tr = X_valid[:split_idx], y_valid[:split_idx]
        X_ts, y_ts = X_valid[split_idx:], y_valid[split_idx:]
        esn = esn_regression(
            size_reservoir=size_r,
            spectral_radius=args[0],
            sparsity=sparsity,
            leaking_rate=args[1],
            input_scaling=args[2],
            reg_param=args[3],
            verbose=0,
            **params
        )
        e_tr = esn.fit(X_tr, y_tr)
        e_ts = esn.evaluate(esn.generate(len(X_ts)), y_ts)
        e_mean += e_ts
    e_mean /= len(valid_indices)
    return (args[0], args[1], args[2], args[3]), e_mean

def randomSearch_multi(X, y, len_valid, k=1, steps=100, size_r=100, sparsity=0.95, sr=None, lr=None, scale_i=None, reg_param=None, params={}):
    # range of search
    param_range = {
        "sr" : list(np.arange(0.05, 3.0, 0.05)) if sr is None else [sr],
        "lr" : list(np.arange(0.0, 1.0, 0.01)) if lr is None else [lr],
        "scale_i" : (1e-3, 1e-2, 1e-1, 1e0, 1e1) if scale_i is None else [scale_i],
        "reg_param": (1e-2, 1e-1, 1e0, 1e1, 1e2) if reg_param is None else [reg_param],
	}

    valid_indices = np.random.choice(len(X)-len_valid-1, k, replace=False).tolist()
    
    compute_func = partial(randomSearch_task, X=X, y=y, len_valid=len_valid, valid_indices=valid_indices, size_r=size_r, sparsity=sparsity, params=params)
    sampled_params = []
    i = 0
    while i < steps:
        sr = random.choice(param_range["sr"])
        lr = random.choice(param_range["lr"])
        scale_i = random.choice(param_range["scale_i"])
        reg_param = random.choice(param_range["reg_param"])
        
        if (sr, lr, scale_i, reg_param) not in sampled_params:
            sampled_params.append((sr, lr, scale_i, reg_param))
            
            i += 1
        else:
            continue
    

    # multiprocessing
    pool = Pool(processes=multiprocessing.cpu_count())
    results = pool.map_async(compute_func, sampled_params)
    pool.close()
    pool.join()
    # find best params and acc
    min_e = np.inf
    best_params = {}
    for result in results.get():
        (sr, lr, scale_i, reg_param), e = result
        if e < min_e:
            min_e = e
            best_params = {
                "sr": sr,
                "lr": lr,
                "scale_i": scale_i,
                "reg_param": reg_param
            }
    print("Best params: {}, error: {}".format(best_params, min_e))
    return best_params, min_e

def read_data_mg():
    data = np.load("./datasets/mg_t17/mackey_glass_t17.npy")
    data = data.reshape(-1, 1) # (10000, 1)

    len_train = 5000
    len_test = 1000

    seq_exp_u = data[:len_train+len_test] # discard the last one
    seq_exp_y = data[1:len_train+len_test+1] # shift one step to predict the next value

    idx_train = len_train
    idx_test = idx_train + len_test


    train_set_u = seq_exp_u[:idx_train]
    train_set_y = seq_exp_y[:idx_train]

    test_set_u = seq_exp_u[idx_train:idx_test]
    test_set_y = seq_exp_y[idx_train:idx_test]

    return train_set_u, train_set_y, test_set_u, test_set_y

def task_searchParams_mg(set_u, set_y, len_valid, k=1, steps=100, params_search={}, params_fixed={}):
    best_params, min_e = randomSearch(
        set_u,
        set_y,
        len_valid=len_valid,
        k=k,
        steps=steps,
        size_r=params_search["size_r"],
        sparsity=params_search["sparsity"],
        sr=params_search["sr"],
        lr=params_search["lr"],
        scale_i=params_search["scale_i"],
        reg_param=params_search["reg_param"],
        params=params_fixed
    )
    return best_params, min_e

def task_train_test_mg(train_set_u, train_set_y, test_set_u, params={}):

    esn = esn_regression(
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
    e_train = esn.fit(train_set_u, train_set_y)
    print("RMSE in training set: {}".format(e_train))

    preds = esn.generate(len(test_set_u))
    trues = test_set_u
    e_test = esn.evaluate(preds, trues)
    print("RMSE in prediction: {}".format(e_test))
    return esn.rout_solver.W.copy(), preds, trues, e_train, e_test

def read_data_lorenz():
    data = np.load("./datasets/lorenz/lorenz_full.npy") # (10000, 3)

    len_train = 5000
    len_test = 1000

    seq_exp_u = data[:len_train+len_test] # discard the last one
    seq_exp_y = data[1:len_train+len_test+1] # shift one step to predict the next value

    idx_train = len_train
    idx_test = idx_train + len_test


    train_set_u = seq_exp_u[:idx_train]
    train_set_y = seq_exp_y[:idx_train]

    test_set_u = seq_exp_u[idx_train:idx_test]
    test_set_y = seq_exp_y[idx_train:idx_test]

    return train_set_u, train_set_y, test_set_u, test_set_y

def task_searchParams_lorenz(set_u, set_y, len_valid, k=1, steps=100, params_search={}, params_fixed={}):
    best_params, min_e = randomSearch(
        set_u,
        set_y,
        len_valid=len_valid,
        k=k,
        steps=steps,
        size_r=params_search["size_r"],
        sparsity=params_search["sparsity"],
        sr=params_search["sr"],
        lr=params_search["lr"],
        scale_i=params_search["scale_i"],
        reg_param=params_search["reg_param"],
        params=params_fixed
    )

    return best_params, min_e

def task_train_test_lorenz(train_set_u, train_set_y, test_set_u, params={}):
    esn = esn_regression(
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
    e_train = esn.fit(train_set_u, train_set_y)
    print("RMSE in training set: {}".format(e_train))

    preds = esn.generate(len(test_set_u))
    trues = test_set_u
    e_test = esn.evaluate(preds, trues)
    print("RMSE in prediction: {}".format(e_test))
    return esn.rout_solver.W.copy(), preds, trues, e_train, e_test

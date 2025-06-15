import os
import sys
sys.path.append(os.path.join(os.getcwd(), 'L1General_python'))
import random

import numpy as np
import pandas as pd
import scipy.io
import matplotlib.pyplot as plt

# esn model
from models.esn_model import esn_classification

# multiprocessing
from functools import partial
import multiprocessing
from multiprocessing import Pool

def randomSearch(X, y, len_valid, k=1, steps=100, size_r=100, sparsity=0.95, sr=None, lr=None, scale_i=None, reg_param=None, params={}):
    # range of search
    param_range = {
        "sr" : list(np.arange(0.05, 3.0, 0.05)) if sr is None else [sr],
        "lr" : list(np.arange(0.0, 1.0, 0.01)) if lr is None else [lr],
        "scale_i" : (1e-3, 1e-2, 1e-1, 1e0, 1e1) if scale_i is None else [scale_i],
        "reg_param": (1e-2, 1e-1, 1e0, 1e1, 1e2) if reg_param is None else [reg_param],
	}
    max_acc = 0
    best_params = {}

    # make fixed length validation set
    X = X[:len_valid*k]
    y = y[:len_valid*k]

    valid_set = []
    for i in range(k):
        idx = i*len_valid
        train_part_X = X[idx:idx+len_valid]
        train_part_y = y[idx:idx+len_valid]
        if i != 0 and i != k-1:
            test_part_X = np.concatenate((X[:idx], X[idx+len_valid:]))
            test_part_y = np.concatenate((y[:idx], y[idx+len_valid:]))
        elif i == 0:
            test_part_X = X[idx+len_valid:]
            test_part_y = y[idx+len_valid:]
        else:
            test_part_X = X[:idx]
            test_part_y = y[:idx]
        valid_set.append((train_part_X, train_part_y, test_part_X, test_part_y))
    
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
        acc_mean = 0.0
        for X_tr, y_tr, X_ts, y_ts in valid_set:
            esn = esn_classification(
                size_reservoir=size_r,
                spectral_radius=sr,
                sparsity=sparsity,
                leaking_rate=lr,
                input_scaling=scale_i,
                reg_param=reg_param,
                verbose=0,
                **params
            )
            acc_tr = esn.fit(X_tr, y_tr)
            acc_ts = esn.evaluate(esn.predict(X_ts), y_ts)
            acc_mean += acc_ts
        acc_mean /= k
        if acc_mean > max_acc:
            max_acc = acc_mean
            best_params = {
                "sr": sr,
                "lr": lr,
                "scale_i": scale_i,
                "reg_param": reg_param
            }
            print("Best params: {}, Best acc: {}".format(best_params, max_acc))
            if acc_mean == 1.0:
                break
    return best_params, max_acc

def randomSearch_task(args, valid_set, size_r=100, sparsity=0.95, params={}):
    acc_mean = 0.0
    for X_valid, y_valid in valid_set:
        split_idx = int(len(X_valid)*0.8)
        X_tr, y_tr = X_valid[:split_idx], y_valid[:split_idx]
        X_ts, y_ts = X_valid[split_idx:], y_valid[split_idx:]
        esn = esn_classification(
            size_reservoir=size_r,
            spectral_radius=args[0],
            sparsity=sparsity,
            leaking_rate=args[1],
            input_scaling=args[2],
            reg_param=args[3],
            verbose=0,
            **params
        )
        acc_tr = esn.fit(X_tr, y_tr)
        acc_ts = esn.evaluate(esn.predict(X_ts), y_ts)
        acc_mean += acc_ts
    acc_mean /= len(valid_set)
    return (args[0], args[1], args[2], args[3]), acc_mean

def randomSearch_multi(X, y, len_valid, k=1, steps=100, size_r=100, sparsity=0.95, sr=None, lr=None, scale_i=None, reg_param=None, params={}):
    # range of search
    param_range = {
        "sr" : list(np.arange(0.05, 3.0, 0.05)) if sr is None else [sr],
        "lr" : list(np.arange(0.0, 1.0, 0.01)) if lr is None else [lr],
        "scale_i" : (1e-3, 1e-2, 1e-1, 1e0, 1e1) if scale_i is None else [scale_i],
        "reg_param": (1e-2, 1e-1, 1e0, 1e1, 1e2) if reg_param is None else [reg_param],
	}

    # make fixed length validation set
    X = X[:len_valid*k]
    y = y[:len_valid*k]

    valid_set = []
    for i in range(k):
        idx = i*len_valid
        valid_set.append((X[idx:idx+len_valid], y[idx:idx+len_valid]))
    
    compute_func = partial(randomSearch_task, valid_set=valid_set, size_r=size_r, sparsity=sparsity, params=params)
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
    

    
    pool = Pool(processes=multiprocessing.cpu_count())
    results = pool.map_async(compute_func, sampled_params)
    pool.close()
    pool.join()
	# find best params and acc
    max_acc = 0
    best_params = {}
    for result in results.get():
        (sr, lr, scale_i, reg_param), acc = result
        if acc > max_acc:
            max_acc = acc
            best_params = {
                "sr": sr,
                "lr": lr,
                "scale_i": scale_i,
                "reg_param": reg_param,
            }
    print("Best params: {}, Best acc: {}".format(best_params, max_acc))
    return best_params, max_acc

def preprocess_written_char(data_dir, fixed_len=128, num_classes=20, shuffle_seqs=True):
    """Preprocess the UCI written character dataset."""

    data = scipy.io.loadmat(data_dir)
    mixout = data["mixout"][0]
    consts = data["consts"][0,0]

    # acquire labels
    key = [item[0] for item in consts["key"][0]] # a, b, c,..., z in total 20 chars
    charlabels = consts["charlabels"][0] - 1 # 1, 1, 1, ... ,20, 20 in total 20 classes

    # padding the seqs
    padded_seqs = np.zeros((len(mixout), fixed_len, 3))
    for i in range(len(mixout)):
        seq = mixout[i].T
        seq_len = len(seq)
        if seq_len < fixed_len:
            padded_seqs[i, :seq_len, :] = seq
        else:
            padded_seqs[i] = seq[:fixed_len,:]
    mixout = padded_seqs
    
    # use previous num_classes characters
    if num_classes < 20 and num_classes > 0:
        classStartIdx_P1 = []
        classStartIdx_P1.append(0)
        classStartIdx_P2 = []
        cntClasses = 0
        maxClasses = 20
        for i in range(1, len(charlabels)):
            if charlabels[i] != charlabels[i-1]:
                cntClasses += 1
                if cntClasses < maxClasses:
                    classStartIdx_P1.append(i)
                else:
                    classStartIdx_P2.append(i)
        new_charlabels = np.hstack([charlabels[classStartIdx_P1[0]:classStartIdx_P1[num_classes]],
                      charlabels[classStartIdx_P2[0]:classStartIdx_P2[num_classes]]])
        new_mixout = np.vstack([mixout[classStartIdx_P1[0]:classStartIdx_P1[num_classes]],
                      mixout[classStartIdx_P2[0]:classStartIdx_P2[num_classes]]])
        charlabels = new_charlabels
        mixout = new_mixout
        key = key[:num_classes]
    elif num_classes == 20:
        charlabels = charlabels
        mixout = mixout
        key = key
    else:
        raise ValueError("num_classes should be in [1, 20]")

    if shuffle_seqs:
        randomized_idx = np.random.permutation(len(mixout))
        mixout = mixout[randomized_idx]
        charlabels = charlabels[randomized_idx]

    return mixout, charlabels, key, data["consts"]
    
def train_test_split(data, labels, test_size=0.2, shuffle=False):
    """Split the data into train and test sets."""
    if shuffle:
        randomized_idx = np.random.permutation(len(data))
        data = data[randomized_idx]
        labels = labels[randomized_idx]

    split_idx = int(len(data) * (1 - test_size))
    train_data = data[:split_idx]
    train_labels = labels[:split_idx]
    test_data = data[split_idx:]
    test_labels = labels[split_idx:]

    return train_data, train_labels, test_data, test_labels

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
    # mean = X_train.mean(axis=(0,1))  # calculate mean along sample and timesteps
    # std = X_train.std(axis=(0,1))    # calculate std along sample and timesteps
    # X_train = (X_train - mean) / (std + 1e-8) 
    # X_test = (X_test - mean) / (std + 1e-8)

    # process labels (original labels are 1-6)
    def load_labels(subset):
        label_path = os.path.join(data_dir, subset, f'y_{subset}.txt')
        labels = np.loadtxt(label_path).astype(int) - 1  # convert to 0-5
        return labels
    # return np.eye(6)[labels]  # convert to one-hot encoding

    y_train = load_labels('train')
    y_test = load_labels('test')


    return X_train, y_train, X_test, y_test



def load_classification_Uni_ts(dataset_name):
    from aeon.datasets import load_from_ts_file
    root_path = "./Univariate_ts/"

    X_train, y_train = load_from_ts_file(root_path+dataset_name+"/"+dataset_name+"_TRAIN.ts", return_type="numpy3d")
    X_test, y_test = load_from_ts_file(root_path+dataset_name+"/"+dataset_name+"_TEST.ts", return_type="numpy3d")
    # switch dimension position
    X_train = np.swapaxes(X_train, 1, 2)
    X_test = np.swapaxes(X_test, 1, 2)
    # y
    if dataset_name in ["DistalPhalanxOutlineCorrect"]:
        y_train = y_train.astype(int)
        y_test = y_test.astype(int)
    else:
        y_train = y_train.astype(int)-1
        y_test = y_test.astype(int)-1
	
    return X_train, y_train, X_test, y_test

def load_classification_Multi_ts(dataset_name, fixed_length=128):
    from aeon.datasets import load_from_ts_file
    root_path = "./Multivariate_ts/"
    X_train, y_train = load_from_ts_file(root_path+dataset_name+"/"+dataset_name+"_TRAIN.ts", return_type="numpy3d")
    X_test, y_test = load_from_ts_file(root_path+dataset_name+"/"+dataset_name+"_TEST.ts", return_type="numpy3d")
    # make X_train and X_test to fixed length
    len_train = len(X_train)
    len_test = len(X_test)
    dim = X_train[0].shape[0]
    padding_X_train = np.zeros((len_train, fixed_length, dim))
    padding_X_test = np.zeros((len_test, fixed_length, dim))
    for i in range(len_train):
        seq = X_train[i].T
        len_seq = seq.shape[0]
        if len_seq < fixed_length:
            diff = fixed_length - len_seq
            padding_X_train[i, diff:, :] = seq[-1]
        else:
            padding_X_train[i] = seq[:fixed_length,:]
    
    for i in range(len_test):
        seq = X_test[i].T
        len_seq = seq.shape[0]
        if len_seq < fixed_length:
            diff = fixed_length - len_seq
            padding_X_test[i, diff:, :] = seq[-1]
        else:
            padding_X_test[i] = seq[:fixed_length,:]
    
    X_train = padding_X_train
    X_test = padding_X_test
    # y
    y_train = y_train.astype(int)-1
    y_test = y_test.astype(int)-1
    return X_train, y_train, X_test, y_test

def read_data_har():
    datapath="./datasets/har"
    X_train, y_train, X_test, y_test = preprocess_uci_har(datapath)

    # get index of random samples
    train_idx = np.random.permutation(X_train.shape[0])
    test_idx = np.random.permutation(X_test.shape[0])

    # select subset of data
    X_train = X_train[train_idx]
    y_train = y_train[train_idx]
    X_test = X_test[test_idx]
    y_test = y_test[test_idx]

    return X_train, y_train, X_test, y_test

def task_searchParams_har(X, y, len_valid, k, steps, params_search, params_fixed={}):
    best_params, acc = randomSearch_multi(
        X=X,
        y=y,
        len_valid=len_valid,
        k=k,
        steps=steps,
        size_r=params_search["size_r"],
        sparsity=params_search["sparsity"],
        sr=params_search["sr"],
        lr=params_search["lr"],
        scale_i=params_search["scale_i"],
        reg_param=params_search["reg_param"],
        params=params_fixed,
    )
    return best_params, acc

def task_train_test_har(X_train, y_train, X_test, y_test, params={}):
    
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
    acc_train = esn.fit(X_train, y_train)
    print("ACC in training set: {}".format(acc_train))

    y_pred = esn.predict(X_test)
    acc_test = esn.evaluate(y_pred, y_test)
    print("ACC in test set: {}".format(acc_test))

    return esn.rout_solver.W.copy(), acc_train, acc_test

def read_data_char():
    datapath="./datasets/char_trajectories/mixoutALL_shifted.mat"
    num_classes = 20
    X, y, _, _ = preprocess_written_char(datapath, num_classes=num_classes)

    # shuffle data
    X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

    return X_train, y_train, X_test, y_test

def task_searchParams_char(X, y, len_valid, k, steps, params_search, params_fixed={}):
    
    best_params, acc = randomSearch_multi(
        X=X,
        y=y,
        len_valid=len_valid,
        k=k,
        steps=steps,
        size_r=params_search["size_r"],
        sparsity=params_search["sparsity"],
        sr=params_search["sr"],
        lr=params_search["lr"],
        scale_i=params_search["scale_i"],
        reg_param=params_search["reg_param"],
        params=params_fixed,
    )
    # print(best_params, acc)
    return best_params, acc


def task_train_test_char(X_train, y_train, X_test, y_test, params={}):
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
    acc_train = esn.fit(X_train, y_train)
    print("ACC in training set: {}".format(acc_train))

    y_pred = esn.predict(X_test)
    acc_test = esn.evaluate(y_pred, y_test)
    print("ACC in test set: {}".format(acc_test))
    return esn.rout_solver.W.copy(), acc_train, acc_test

def read_data_Uni(dataset_name):
    X_train, y_train, X_test, y_test = load_classification_Uni_ts(dataset_name)

    train_idx = np.random.permutation(X_train.shape[0])
    test_idx = np.random.permutation(X_test.shape[0])

    X_train = X_train[train_idx]
    y_train = y_train[train_idx]
    X_test = X_test[test_idx]
    y_test = y_test[test_idx]

    return X_train, y_train, X_test, y_test

def task_searchParams_Uni(X, y, len_valid, k, steps, params_search, params_fixed={}):
    best_params, acc = randomSearch_multi(
        X=X,
        y=y,
        len_valid=len_valid,
        k=k,
        steps=steps,
        size_r=params_search["size_r"],
        sparsity=params_search["sparsity"],
        sr=params_search["sr"],
        lr=params_search["lr"],
        scale_i=params_search["scale_i"],
        reg_param=params_search["reg_param"],
        params=params_fixed,
    )
    # print(best_params, acc)
    return best_params, acc

def task_train_test_Uni(X_train, y_train, X_test, y_test, params={}):
    
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
    acc_train = esn.fit(X_train, y_train)
    print("ACC in training set: {}".format(acc_train))

    y_pred = esn.predict(X_test)
    acc_test = esn.evaluate(y_pred, y_test)
    print("ACC in test set: {}".format(acc_test))
    return esn.rout_solver.W.copy(), acc_train, acc_test
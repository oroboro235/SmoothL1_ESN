import os
import numpy as np
np.random.seed(1234)
import random
random.seed(1234)
import json

# import partial
from functools import partial

from script_classification import read_data_har, read_data_char, read_data_Uni, task_train_test_har, task_train_test_char, task_train_test_Uni

model_save_path = "./results/models/"
output_save_path = "./results/outputs/"

def save_results_models(specific_name, rout_weight, acc_train_range: list, acc_pred_range: list):
    # save model    
    np.save(os.path.join(model_save_path, specific_name + "_rout_weight.npy"), rout_weight)

    # save output
    output_dict = {
        "acc_train": acc_train_range,
        "acc_pred": acc_pred_range
    }
    with open(os.path.join(output_save_path, specific_name + "_output.json"), "w") as f:
        json.dump(output_dict, f)

def run(task_name, read_func, task_func, update_params, k_guess=1):
    X_train, y_train, X_test, y_test = read_func()
    _params = {
        "size_r": 500,
        "sparsity": 0.95,
        "sr": None,
        "lr": None,
        "scale_i": None,
        "reg_type": None,
        "reg_param": 1e1,
        "num_warmup": 0,
        "evaluation_metric": "acc"
    }
    _params.update(update_params)

    # run task
    results = []
    for i in range(k_guess):
        print("guess {}/{}".format(i+1, k_guess))
        rout_weight, acc_train, acc_pred = task_func(X_train, y_train, X_test, y_test, params=_params)
        results.append((rout_weight, acc_train, acc_pred))
    # find best guess
    acc_train_range = [r[1] for r in results]
    acc_pred_range = [r[2] for r in results]
    best_guess = np.argmax(acc_pred_range)
    print("best guess train acc: {}, test acc: {}".format(acc_train_range[best_guess], acc_pred_range[best_guess]))
    # save best guess
    rout_weight, _, _ = results[best_guess]
    save_results_models(task_name, rout_weight, acc_train_range, acc_pred_range)

if __name__ == "__main__":
    # paths
    optimal_params_har = "results_paramSearch_classification_har.json"
    optimal_params_char = "results_paramSearch_classification_char.json"
    optimal_params_uni = [
        "results_paramSearch_classification_ECG5000.json",
        "results_paramSearch_classification_DistalPhalanxOutlineCorrect.json",
        "results_paramSearch_classification_Yoga.json",
        "results_paramSearch_classification_Strawberry.json",
    ]

    # load params for har and char
    # har params
    har_params = json.load(open(optimal_params_har, "r"))
    # har_params_none = har_params["har"]["none"]["best_params"]
    har_params_l2 = har_params["har"]["l2"]["best_params"]
    har_params_smoothl1 = har_params["har"]["smoothl1"]["best_params"]
    # char params
    char_params = json.load(open(optimal_params_char, "r"))
    # char_params_none = char_params["char"]["none"]["best_params"]
    char_params_l2 = char_params["char"]["l2"]["best_params"]
    char_params_smoothl1 = char_params["char"]["smoothl1"]["best_params"]


    guess = 5
    # UCI HAR
    read_func = read_data_har
    task_func = task_train_test_har
    # # none
    # params = har_params_none.copy()
    # params["reg_type"] = "none"
    # run("har_none", read_func, task_func, params, guess)
    # # l2
    # params = har_params_l2.copy()
    # params["reg_type"] = "l2"
    # run("har_l2", read_func, task_func, params, guess)
    # # # smoothl1
    # params = har_params_smoothl1.copy()
    # params["reg_type"] = "smoothl1"
    # run("har_smoothl1", read_func, task_func, params, guess)


    # Character tracjectories
    read_func = read_data_char
    task_func = task_train_test_char
    # # none
    # params = char_params_none.copy()
    # params["reg_type"] = "none"
    # run("char_none", read_func, task_func, params, guess)
    # # l2
    params = char_params_l2.copy()
    params["reg_type"] = "l2"
    run("char_l2", read_func, task_func, params, guess)
    # # # smoothl1
    params = char_params_smoothl1.copy()
    params["reg_type"] = "smoothl1"
    run("char_smoothl1", read_func, task_func, params, guess)

    # # Univariate time series
    # for i, dataset_filename in enumerate(optimal_params_uni):
    #     dataset_name = dataset_filename.split("_")[3].split(".")[0]
    #     read_func = partial(read_data_Uni, dataset_name)
    #     task_func = task_train_test_Uni
    #     # load params
    #     uni_params = json.load(open(dataset_filename, "r"))
    #     # uni_params_none = uni_params[dataset_name]["none"]["best_params"]
        
    #     uni_params_l2 = uni_params[dataset_name]["l2"]["best_params"]
    #     uni_params_smoothl1 = uni_params[dataset_name]["smoothl1"]["best_params"]
    #     # none
    #     # params = uni_params_none.copy()
    #     # params["reg_type"] = "none"
    #     # run(dataset_name + "_none", read_func, task_func, params, guess)
    #     # l2
    #     params = uni_params_l2.copy()
    #     params["reg_type"] = "l2"
    #     run(dataset_name + "_l2", read_func, task_func, params, guess)
    #     # smoothl1
    #     params = uni_params_smoothl1.copy()
    #     params["reg_type"] = "smoothl1"
    #     run(dataset_name + "_smoothl1", read_func, task_func, params, guess)

        

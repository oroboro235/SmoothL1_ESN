import os
import numpy as np
np.random.seed(1234)
import random
random.seed(1234)
import json

from functools import partial

from script_classification import task_train_test_har, task_train_test_char, task_train_test_Uni, read_data_har, read_data_char, read_data_Uni



def test_lambda(range_lambda:list, read_func, task_func, update_params, k_runs=1):
    X_train, y_train, X_test, y_test = read_func()
    _params = {
        "size_r": 500,
        "sparsity": 0.95,
        "sr": None,
        "lr": None,
        "scale_i": None,
        "reg_type": "smoothl1",
        "reg_param": 1e1,
        "num_warmup": 0,
        "evaluation_metric": "acc",
    }
    # update params
    _params.update(update_params)
    results = []
    for _lambda in range_lambda:
        _params["reg_param"] = _lambda
        for i in range(k_runs):
            print("lambda: {}, run: {}".format(_lambda, i))
            _, _, acc_test = task_func(X_train, y_train, X_test, y_test, params=_params)  
            results.append((_lambda, acc_test))
    return results

def test_size_r(range_size:list, read_func, task_func, update_params, k_runs=1):
    X_train, y_train, X_test, y_test = read_func()
    _params = {
        "size_r": 500,
        "sparsity": 0.95,
        "sr": None,
        "lr": None,
        "scale_i": None,
        "reg_type": "smoothl1",
        "reg_param": 1e1,
        "num_warmup": 0,
        "evaluation_metric": "acc",
    }
    # update params
    _params.update(update_params)
    results = []
    for size_r in range_size:
        _params["size_r"] = size_r
        for i in range(k_runs):
            print("size_r: {}, run: {}".format(size_r, i))
            _, _, acc_test = task_func(X_train, y_train, X_test, y_test, params=_params)  
            results.append((size_r, acc_test))
        
    return results

def save_results(exp_name, results, save_path="./"):
    results_dict = {}
    for result in results:
        if result[0] not in results_dict:
            results_dict[result[0]] = []
            results_dict[result[0]].append(result[1])
        else:
            results_dict[result[0]].append(result[1])
    with open(os.path.join(save_path, exp_name + ".json"), 'w') as f:
        json.dump(results_dict, f)


if __name__ == '__main__':
    # har and char paths
    optimal_har_params_file = "results_paramSearch_classification_har.json"
    optimal_char_params_file = "results_paramSearch_classification_char.json"
    # read optimal parameters from files
    har_params = json.load(open(optimal_har_params_file, 'r'))
    char_params = json.load(open(optimal_char_params_file, 'r'))

    har_params_smoothl1 = har_params["har"]["smoothl1"]["best_params"]
    char_params_smoothl1 = char_params["char"]["smoothl1"]["best_params"]


    # run test for different lambda
    # range_lambda = (1e-2, 1e-1, 1, 10, 100)
    # # results_lambda_mg = test_lambda(range_lambda, read_data_mg, task_train_test_mg, mg_params_smoothl1, k_runs=3)
    # results_lambda_har = test_lambda(range_lambda, read_data_har, task_train_test_har, har_params_smoothl1, k_runs=3)
    # save_results("results_lambda_har", results_lambda_har, "./results/different_params/")

    # # run test for different size_r
    # range_size_r = (100, 200, 500, 1000)
    # results_size_r_har = test_size_r(range_size_r, read_data_har, task_train_test_har, har_params_smoothl1, k_runs=3)
    # save_results("results_size_r_har", results_size_r_har, "./results/different_params/")

    # run test for different lambda
    # range_lambda = (1e-2, 1e-1, 1, 10, 100)
    # results_lambda_char = test_lambda(range_lambda, read_data_char, task_train_test_char, char_params_smoothl1, k_runs=3)   
    # save_results("resulst_lambda_char", results_lambda_char, "./results/different_params/")

    # run test for different size_r
    # range_size_r = (100, 200, 500, 1000)
    # results_size_r_char = test_size_r(range_size_r, read_data_char, task_train_test_char, char_params_smoothl1, k_runs=3)
    # save_results("results_size_r_char", results_size_r_char, "./results/different_params/")

    # univariate
    optimal_params_uni = [
        "results_paramSearch_classification_ECG5000.json",
        "results_paramSearch_classification_DistalPhalanxOutlineCorrect.json",
        "results_paramSearch_classification_Yoga.json",
        "results_paramSearch_classification_Strawberry.json",
    ]
    for i, dataset_filename in enumerate(optimal_params_uni):
        dataset_name = dataset_filename.split("_")[3].split(".")[0]
        read_func = partial(read_data_Uni, dataset_name)
        task_func = task_train_test_Uni
        # load params
        uni_params = json.load(open(dataset_filename, "r"))
        uni_params_smoothl1 = uni_params[dataset_name]["smoothl1"]["best_params"]

        # smoothl1
        # lambda
        range_lambda = (1e-2, 1e-1, 1, 10, 100)
        results_lambda_uni = test_lambda(range_lambda, read_func, task_func, uni_params_smoothl1, k_runs=3)
        save_results("results_lambda_uni_{}".format(dataset_name), results_lambda_uni, "./results/different_params/")

        # size_r
        range_size_r = (100, 200, 500, 1000)
        results_size_r_uni = test_size_r(range_size_r, read_func, task_func, uni_params_smoothl1, k_runs=3)
        save_results("results_size_r_uni_{}".format(dataset_name), results_size_r_uni, "./results/different_params/")



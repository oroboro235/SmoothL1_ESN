import os
import numpy as np
np.random.seed(1234)
import random
random.seed(1234)
import json


from script_regression import task_train_test_mg, task_train_test_lorenz, read_data_mg, read_data_lorenz



def test_lambda(range_lambda:list, read_func, task_func, update_params, k_runs=1):
    train_set_u, train_set_y, test_set_u, test_set_y = read_func()
    _params = {
        "size_r": 500,
        "sparsity": 0.95,
        "sr": None,
        "lr": None,
        "scale_i": None,
        "reg_type": "smoothl1",
        "reg_param": 1e-1,
        "num_warmup": 100,
        "evaluation_metric": "rmse",
    }
    # update params
    _params.update(update_params)
    results = []
    for _lambda in range_lambda:
        _params["reg_param"] = _lambda
        for i in range(k_runs):
            print("lambda: {}, run: {}".format(_lambda, i))
            _, _, _, _, e_test = task_func(train_set_u, train_set_y, test_set_u, params=_params)  
            results.append((_lambda, e_test))
    return results

def test_size_r(range_size:list, read_func, task_func, update_params, k_runs=1):
    train_set_u, train_set_y, test_set_u, test_set_y = read_func()
    _params = {
        "size_r": 500,
        "sparsity": 0.95,
        "sr": None,
        "lr": None,
        "scale_i": None,
        "reg_type": "smoothl1",
        "reg_param": 1e-1,
        "num_warmup": 100,
        "evaluation_metric": "rmse",
    }
    # update params
    _params.update(update_params)
    results = []
    for size_r in range_size:
        _params["size_r"] = size_r
        for i in range(k_runs):
            print("size_r: {}, run: {}".format(size_r, i))
            _, _, _, _, e_test = task_func(train_set_u, train_set_y, test_set_u, params=_params)  
            results.append((size_r, e_test))
        
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
    # paths
    # optimal_params_file = "results_paramSearch_regression.json"
    optimal_mg_params_file = "results_paramSearch_regression_mg.json"
    optimal_lorenz_params_file = "results_paramSearch_regression_lorenz.json"
    # read optimal parameters from files
    # regression_params = json.load(open(optimal_params_file, 'r'))
    mg_params = json.load(open(optimal_mg_params_file, 'r'))
    lorenz_params = json.load(open(optimal_lorenz_params_file, 'r'))

    mg_params_smoothl1 = mg_params["mg"]["smoothl1"]["best_params"]
    lorenz_params_smoothl1 = lorenz_params["lorenz"]["smoothl1"]["best_params"]

    # run test for different lambda
    range_lambda = (1e-2, 1e-1, 1, 10, 100)
    results_lambda_mg = test_lambda(range_lambda, read_data_mg, task_train_test_mg, mg_params_smoothl1, k_runs=3)
    save_results("results_lambda_mg", results_lambda_mg, "./results/different_params/")

    # run test for different size_r
    range_size_r = (100, 200, 500, 1000, 2000)
    results_size_r_mg = test_size_r(range_size_r, read_data_mg, task_train_test_mg, mg_params_smoothl1, k_runs=3)
    save_results("results_size_r_mg", results_size_r_mg, "./results/different_params/")

    # run test for different lambda
    range_lambda = (1e-2, 1e-1, 1, 10, 100)
    results_lambda_lorenz = test_lambda(range_lambda, read_data_lorenz, task_train_test_lorenz, lorenz_params_smoothl1, k_runs=3)
    save_results("results_lambda_lorenz", results_lambda_lorenz, "./results/different_params/")

    # run test for different size_r
    range_size_r = (100, 200, 500, 1000, 2000)
    results_size_r_lorenz = test_size_r(range_size_r, read_data_lorenz, task_train_test_lorenz, lorenz_params_smoothl1, k_runs=3)
    save_results("results_size_r_lorenz", results_size_r_lorenz, "./results/different_params/")



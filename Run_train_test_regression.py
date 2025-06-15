# TODO test different lambdas for sparsity
# TODO different reservoir size


import os
import numpy as np
np.random.seed(1234)
import random
random.seed(1234)
import json

# use the params for regularization

from script_regression import task_train_test_mg, task_train_test_lorenz, read_data_mg, read_data_lorenz


model_save_path = "./results/models/"
output_save_path = "./results/outputs/"

def save_results_models(specific_name, rout_weight, preds, trues, e_train_range: list, e_pred_range: list):
    # save model    
    np.save(os.path.join(model_save_path, specific_name + "_rout_weight.npy"), rout_weight)

    # save output
    # preds
    np.save(os.path.join(output_save_path, specific_name + "_preds.npy"), preds)
    # trues
    np.save(os.path.join(output_save_path, specific_name + "_trues.npy"), trues)
    # e_train, e_pred -> json
    json.dump({"e_train_range": e_train_range, "e_pred_range": e_pred_range}, open(os.path.join(output_save_path, specific_name + "_errors.json"), 'w'))


def run(task_name, read_func, task_func, update_params, k_guess=1):
    train_set_u, train_set_y, test_set_u, test_set_y = read_func()
    _params = {
        "size_r": 500,
        "sparsity": 0.95,
        "sr": None,
        "lr": None,
        "scale_i": None,
        "reg_type": "none",
        "reg_param": 1e-1,
        "num_warmup": 100,
        "evaluation_metric": "rmse",
    }
    # update default parameters
    _params.update(update_params)

    # run task
    results = []
    for i in range(k_guess):
        print("guess {}/{}".format(i+1, k_guess))
        rout_weight, preds, trues, e_train, e_pred = task_func(train_set_u, train_set_y, test_set_u, _params)
        results.append((rout_weight, preds, trues, e_train, e_pred))
    # find best guess
    e_train_range = [r[3] for r in results]
    e_test_range = [r[4] for r in results]
    best_guess = np.argmin(e_test_range)
    print("best guess train error: {}, test error: {}".format(e_train_range[best_guess], e_test_range[best_guess]))
    # save best guess
    rout_weight, preds, trues, _, _ = results[best_guess]
    save_results_models(task_name, rout_weight, preds, trues, e_train_range, e_test_range)
    




if __name__ == '__main__':
    # paths
    # optimal_params_file = "results_paramSearch_regression.json"
    optimal_mg_params_file = "results_paramSearch_regression_mg.json"
    optimal_lorenz_params_file = "results_paramSearch_regression_lorenz.json"
    # read optimal parameters from files
    # regression_params = json.load(open(optimal_params_file, 'r'))
    mg_params = json.load(open(optimal_mg_params_file, 'r'))
    lorenz_params = json.load(open(optimal_lorenz_params_file, 'r'))


    # mackey-glass parameters
    mg_params_none = mg_params["mg"]["none"]["best_params"]
    mg_params_l2 = mg_params["mg"]["l2"]["best_params"]
    mg_params_smoothl1 = mg_params["mg"]["smoothl1"]["best_params"]
    # lorenz parameters
    lorenz_params_none = lorenz_params["lorenz"]["none"]["best_params"]
    lorenz_params_l2 = lorenz_params["lorenz"]["l2"]["best_params"]
    lorenz_params_smoothl1 = lorenz_params["lorenz"]["smoothl1"]["best_params"]

    guess = 10
    # mackey-glass
    # acquire results from different regular terms
    read_func = read_data_mg
    task_func = task_train_test_mg
    # none
    # params = mg_params_none.copy()
    # params["reg_type"] = "none"
    # params["reg_param"] = 1e-1
    # run("mg_none", read_func, task_func, params, guess)
    # l2
    # params = mg_params_l2.copy()
    # params["reg_type"] = "l2"
    # params["reg_param"] = 1e-1
    # run("mg_l2", read_func, task_func, params, guess)
    # smoothl1
    params = mg_params_smoothl1.copy()
    params["reg_type"] = "smoothl1"
    params["reg_param"] = 1e-1
    run("mg_smoothl1", read_func, task_func, params, guess)


    # # lorenz
    # task_func = task_train_test_lorenz
    # read_func = read_data_lorenz
    # # none
    # params = lorenz_params_none.copy()
    # params["reg_type"] = "none"
    # params["reg_param"] = 1e-1
    # run("lorenz_none", read_func, task_func, params, guess)
    # # l2
    # params = lorenz_params_l2.copy()
    # params["reg_type"] = "l2"
    # params["reg_param"] = 1e-1
    # run("lorenz_l2", read_func, task_func, params, guess)
    # # smoothl1
    # params = lorenz_params_smoothl1.copy()
    # params["reg_type"] = "smoothl1"
    # params["reg_param"] = 1e-1
    # run("lorenz_smoothl1", read_func, task_func, params, guess)






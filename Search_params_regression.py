
import json

# import task
from script_regression import task_searchParams_mg, task_searchParams_lorenz, read_data_mg, read_data_lorenz

def task_searchParams_reg_type(read_func, task_func, len_valid=500, k=1, steps=500, params_search={}, params_fixed={}):
    # read data
    train_set_u, train_set_y, _, _ = read_func()
    # none
    params_fixed["reg_type"] = "none"
    best_params_none, min_e_none = task_func(train_set_u, train_set_y, len_valid, k, steps, params_search, params_fixed)
    params_fixed["reg_type"] = "l2"
    best_params_l2, min_e_l2 = task_func(train_set_u, train_set_y, len_valid, k, steps, params_search, params_fixed)
    params_fixed["reg_type"] = "smoothl1"
    best_params_smoothl1, min_e_smoothl1 = task_func(train_set_u, train_set_y, len_valid, k, steps, params_search, params_fixed)
    return best_params_none, min_e_none, best_params_l2, min_e_l2, best_params_smoothl1, min_e_smoothl1

def save_params_to_json(params_none, metric_none, params_l2, metric_l2, params_smoothl1, metric_smoothl1, dataset_name):
    results = {}
    results[dataset_name] = {
        "none": {
            "best_params": params_none,
            "min_e": metric_none
            },
        "l2": {
            "best_params": params_l2,
            "min_e": metric_l2
            },
        "smoothl1": {
            "best_params": params_smoothl1,
            "min_e": metric_smoothl1
            }
        }
    with open("results_paramSearch_regression_"+dataset_name+".json", "w") as f:
        json.dump(results, f)
    print("Saved results to json file: results_paramSearch_regression_"+dataset_name+".json")

if __name__ == '__main__':
    # mackey-glass series parameters searching task
    param_search = {
        "size_r": 100,
        "sparsity": 0.95,
        "sr": None,
        "lr": None,
        "scale_i": None,
        "reg_param": 1.0,
    }
    params_fixed = {
        "reg_type": "none",
        "num_warmup": 100,
        "evaluation_metric": "rmse" 
    }


    # mackey-glass
    params_mg_none, min_e_mg_none, params_mg_l2, min_e_mg_l2, params_mg_smoothl1, min_e_mg_smoothl1 = task_searchParams_reg_type(read_data_mg, task_searchParams_mg, 500, 5, 500, param_search, params_fixed)
    print("Finished searching params for mackey-glass series")
    save_params_to_json(params_mg_none, min_e_mg_none, params_mg_l2, min_e_mg_l2, params_mg_smoothl1, min_e_mg_smoothl1, "mg")

    # lorenz
    params_lorenz_none, min_e_lorenz_none, params_lorenz_l2, min_e_lorenz_l2, params_lorenz_smoothl1, min_e_lorenz_smoothl1 = task_searchParams_reg_type(read_data_lorenz, task_searchParams_lorenz, 500, 5, 500, param_search, params_fixed)
    print("Finished searching params for lorenz series")
    save_params_to_json(params_lorenz_none, min_e_lorenz_none, params_lorenz_l2, min_e_lorenz_l2, params_lorenz_smoothl1, min_e_lorenz_smoothl1, "lorenz")


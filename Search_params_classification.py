
import json

from script_classification import read_data_har, read_data_char, read_data_Uni
from script_classification import task_searchParams_har, task_searchParams_char, task_searchParams_Uni

def task_searchParams_reg_type(read_func, task_func, len_valid=100, k=1, steps=500, params_search={}, params_fixed={}):
    X_train, y_train, _, _ = read_func()
    params_fixed["reg_type"] = "none"
    best_params_none, max_acc_none = task_func(X_train, y_train, len_valid, k, steps, params_search, params_fixed)
    params_fixed["reg_type"] = "l2"
    best_params_l2, max_acc_l2 = task_func(X_train, y_train, len_valid, k, steps, params_search, params_fixed)
    params_fixed["reg_type"] = "smoothl1"
    best_params_smoothl1, max_acc_smoothl1 = task_func(X_train, y_train, len_valid, k, steps, params_search, params_fixed)
    return best_params_none, max_acc_none, best_params_l2, max_acc_l2, best_params_smoothl1, max_acc_smoothl1


def task_searchParams_uni_reg_type(dataset_name, len_valid=100, k=1, steps=500, params_search={}, params_fixed={}):
    X_train, y_train, _, _ = read_data_Uni(dataset_name)
    params_fixed["reg_type"] = "none"
    best_params_none, max_acc_none = task_searchParams_Uni(X_train, y_train, len_valid, k, steps, params_search, params_fixed)
    params_fixed["reg_type"] = "l2"
    best_params_l2, max_acc_l2 = task_searchParams_Uni(X_train, y_train, len_valid, k, steps, params_search, params_fixed)
    params_fixed["reg_type"] = "smoothl1"
    best_params_smoothl1, max_acc_smoothl1 = task_searchParams_Uni(X_train, y_train, len_valid, k, steps, params_search, params_fixed)
    return best_params_none, max_acc_none, best_params_l2, max_acc_l2, best_params_smoothl1, max_acc_smoothl1

def save_params_to_json(params_none, metric_none, params_l2, metric_l2, params_smoothl1, metric_smoothl1, dataset_name):
    results = {}
    results[dataset_name] = {
        "none": {
            "best_params": params_none,
            "max_acc": metric_none
        },
        "l2": {
            "best_params": params_l2,
            "max_acc": metric_l2
        },
        "smoothl1": {
            "best_params": params_smoothl1,
            "max_acc": metric_smoothl1
        }
    }
    with open("results_paramSearch_classification_"+dataset_name+".json", "w") as f:
        json.dump(results, f)
    print("Saved results to json file: results_paramSearch_classification_"+dataset_name+".json")

if __name__ == '__main__':
    # params for searching
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
        "num_warmup": 0,
        "evaluation_metric": "acc" 
    }

    # # HAR
    params_har_none, max_acc_har_none, params_har_l2, max_acc_l2, params_har_smoothl1, max_acc_smoothl1 = task_searchParams_reg_type(read_data_har, task_searchParams_har, 100, 5, 500, params_search=param_search, params_fixed=params_fixed)
    print("Finished searching params for UCI HAR dataset")
    save_params_to_json(params_har_none, max_acc_har_none, params_har_l2, max_acc_l2, params_har_smoothl1, max_acc_smoothl1, "har")


    # # character trajectory
    params_char_none, max_acc_char_none, params_char_l2, max_acc_char_l2, params_char_smoothl1, max_acc_char_smoothl1 = task_searchParams_reg_type(read_data_char,task_searchParams_char, 100, 5, 500, params_search=param_search, params_fixed=params_fixed)
    print("Finished searching params for character trajectory dataset")
    save_params_to_json(params_char_none, max_acc_char_none, params_char_l2, max_acc_char_l2, params_char_smoothl1, max_acc_char_smoothl1, "char")


    # UCR & UEA datasets
    list_uni_datasets = [
        "ECG5000",
        "DistalPhalanxOutlineCorrect",
        "Yoga",
        "Strawberry",
    ]
    for dataset_name in list_uni_datasets:
        print("Running for dataset: {}".format(dataset_name))
        best_params_none, max_acc_none, best_params_l2, max_acc_l2, best_params_smoothl1, max_acc_smoothl1 = task_searchParams_uni_reg_type(dataset_name, 100, 5, 500, params_search=param_search, params_fixed=params_fixed)
        print("Finished searching params for dataset: {}".format(dataset_name))
        save_params_to_json(best_params_none, max_acc_none, best_params_l2, max_acc_l2, best_params_smoothl1, max_acc_smoothl1, dataset_name)

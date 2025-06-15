import numpy as np

from sklearn.linear_model import Ridge, Lasso, LassoLars, LinearRegression
from functools import partial
from L1General_python.lossFuncs import SquaredError, SquaredError_noCupy, softmaxLoss2, softmaxLoss2_noCupy
from L1General_python.L1General import L1GeneralUnconstrainedApx

from scipy.optimize import minimize

# from L1General_python.objFunc import get_loss_func

class readout_linearRegression():
    def __init__(
            self,
            size_i,
            size_o,
            reg_type="l2",
            reg_param=1e-2,
            verbose=1):
        self.size_i = size_i
        self.size_o = size_o
        self.reg_type = reg_type
        self.reg_param = reg_param
        self.verbose = verbose
        self.W = np.zeros((size_i, size_o))
    
    def fit(self, X, y):
        WO = np.zeros((self.size_i, self.size_o))

        if self.reg_type == "none":
            mode = 2
        elif self.reg_type == "l2":
            mode = 1
        elif self.reg_type == "smoothl1":
            mode = 0
        else:
            raise ValueError("Invalid regularization type")

        for i in range(self.size_o):
            WO_init = np.zeros((self.size_i, 1))
            funcObj = partial(SquaredError_noCupy, X=X, y=y[:, i])
            _lambda_vec = self.reg_param * np.ones(self.size_i)
            options = {}
            options['verbose'] = self.verbose
            options['mode'] = mode
            options['progTol'] = 1e-12
            wLR, _ = L1GeneralUnconstrainedApx(funcObj, WO_init, _lambda_vec, options)

            WO[:, i] = wLR.reshape(-1)
        self.W = WO
        return self.W

    def predict(self, X):
        return X @ self.W
    

class readout_logisticRegression():
    def __init__(
            self,
            size_i,
            size_o,
            reg_type="l2",
            reg_param=1e-2,
            verbose=1):
        self.size_i = size_i
        self.size_o = size_o
        self.reg_type = reg_type
        self.reg_param = reg_param
        self.verbose = verbose
        self.W = np.zeros((size_i, size_o))

    def fit(self, X, y):
        # WO = np.zeros((self.size_i, self.size_o))

        if self.reg_type == "none":
            mode = 2
            # loss_f, loss_g, loss_H = get_loss_func(X, y, 'logistic_regression', 'none', self.reg_param, self.size_o)
        elif self.reg_type == "l2":
            mode = 1
            # loss_f, loss_g, loss_H = get_loss_func(X, y, 'logistic_regression', 'l2', self.reg_param, self.size_o)
        elif self.reg_type == "smoothl1":
            mode = 0
            # loss_f, loss_g, loss_H = get_loss_func(X, y, 'logistic_regression','smoothl1', self.reg_param, self.size_o)
        else:
            raise ValueError("Invalid regularization type")

        # size_o is the number of classes
        w_init = np.zeros((self.size_i, self.size_o-1))
        # w_init = np.random.randn(self.size_i, self.size_o-1)
        w_init = w_init.ravel(order='F').reshape(-1, 1)
        funcObj = partial(softmaxLoss2_noCupy, X=X, y=y, k=self.size_o)

        _lambda_vec = self.reg_param * np.ones((self.size_i, self.size_o-1))
        _lambda_vec = _lambda_vec.ravel(order='F').reshape(-1, 1)

        options = {}
        options['verbose'] = self.verbose
        options['mode'] = mode
        options['progTol'] = 1e-12
        # options["optTol"] = 1e-3
        WO, _ = L1GeneralUnconstrainedApx(funcObj, w_init, _lambda_vec, options)
        # WO = minimize(loss_f, w_init.squeeze(), method="BFGS", jac=loss_g, options={"disp":True}).x

        WO = WO.reshape((self.size_i, self.size_o-1), order='F')
        WO = np.concatenate((WO, np.zeros((self.size_i, 1))), axis=1)

        self.W = WO
        return self.W
    
    def predict(self, X):
        logits = X @ self.W
        return np.argmax(logits, axis=1)


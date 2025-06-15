import numpy as np

from models.reservoir import Reservoir
from models.readout import readout_linearRegression, readout_logisticRegression

class esn_regression():
    def __init__(
            self,
            size_reservoir=100,
            spectral_radius=1.0,
            sparsity=0.9,
            leaking_rate=1.0,
            input_scaling=1.0,
            reg_type="l2",
            reg_param=1e-2,
            num_warmup=0,
            evaluation_metric="rmse",
            verbose=1
        ):
        self.res = Reservoir(size_reservoir,
                             spectral_radius,
                             sparsity,
                             leaking_rate,
                             input_scaling)
        self.rout_solver = None
        self.reg_type = reg_type
        self.reg_param = reg_param
        self.n_warmup = num_warmup
        self.eval = evaluation_metric
        self.verbose = verbose

    def fit(self, X, y):
        # acquire the dimension of input
        # X (n_samples, n_dim)
        _, dim = X.shape
        states, last_state = self.res.collect_states(X)
        self.res_last_state = last_state

        states = states[self.n_warmup:]
        y = y[self.n_warmup:]

        # initialize the readout
        self.rout_solver = readout_linearRegression(self.res.size_r,
                                  dim, 
                                  self.reg_type, 
                                  self.reg_param,
                                  self.verbose)
        self.rout_solver.fit(states, y)
        y_pred = self.rout_solver.predict(states)
        return self.evaluate(y_pred, y)

    def generate(self, n_generate=100, update_state=False):
        # generate the predictions in auto-regression way
        # initilize the reservoir state with last state after fitting
        self.res.state_r = self.res_last_state
        preds = []
        preds.append(self.res.state_r @ self.rout_solver.W)
        for _ in range(n_generate-1):
            self.res.forward(preds[-1])
            preds.append(self.res.state_r @ self.rout_solver.W)
        preds = np.array(preds)
        if update_state:
            self.res_last_state = self.res.state_r
        return preds
    
    def evaluate(self, y_pred, y):
        if self.eval == "mse":
            return np.mean((y_pred - y)**2)
        elif self.eval == "rmse":
            return np.sqrt(np.mean((y_pred - y)**2))
        else:
            raise ValueError("Invalid evaluation metric")
        



class esn_classification():
    def __init__(
            self,
            size_reservoir=100,
            spectral_radius=1.0,
            sparsity=0.9,
            leaking_rate=1.0,
            input_scaling=1.0,
            reg_type="l2",
            reg_param=1e-2,
            num_warmup=0,
            evaluation_metric="acc",
            verbose=1,
        ):
        self.res = Reservoir(size_reservoir,
                             spectral_radius,
                             sparsity,
                             leaking_rate,
                             input_scaling)
        self.rout_solver = None
        self.reg_type = reg_type
        self.reg_param = reg_param
        self.n_warmup = num_warmup
        self.eval = evaluation_metric
        self.verbose = verbose

    def get_seq_lstates(self, X):
        # X (n_samples, len_seq, n_dim)
        collect_lstates = []
        for seq in X:
            seq_len = len(seq)
            self.res.reset_state()
            _, last_state = self.res.collect_states(seq)
            collect_lstates.append(last_state)
        X_seq = np.array(collect_lstates)
        return X_seq

    def fit(self, X, y):
        # acquire the dimension of input
        # X (n_samples, len_seq, n_dim)
        n_classes = np.bincount(y).shape[0]
        X_seq = self.get_seq_lstates(X)
        self.rout_solver = readout_logisticRegression(
                                self.res.size_r,
                                n_classes, 
                                self.reg_type, 
                                self.reg_param,
                                self.verbose)

        self.rout_solver.fit(X_seq, y)
        y_pred = self.rout_solver.predict(X_seq)

        return self.evaluate(y_pred, y)

    def predict(self, X):
        X_seq = self.get_seq_lstates(X)
        return self.rout_solver.predict(X_seq)
    
    def evaluate(self, y_pred, y):
        if self.eval == "acc":
            return np.mean(y_pred == y)
        else:
            raise ValueError("Invalid evaluation metric")
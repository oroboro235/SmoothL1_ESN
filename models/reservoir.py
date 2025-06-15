import numpy as np

import scipy.sparse as sp

class Reservoir():
    def __init__(
            self,
            reservoir_size=100,
            spectral_radius=1.0,
            sparsity=0.5,
            leaking_rate=1.0,
            input_scaling=1.0,
            nonlinearity=np.tanh,
        ):
        self.size_i = None
        self.size_r = reservoir_size
        self.sr = spectral_radius
        self.sparsity = sparsity
        self.lr = leaking_rate
        self.scale_i = input_scaling
        self.fn_nonlin = nonlinearity
        self.reset_all()
        
    def reset_all(self):
        self.size_i = None
        self.W_in = None
        self.reset_reservoir_weight()
        self.reset_state()

    def reset_input_weight(self, size_i):
        # initialize the input weight with uniform distribution
        self.W_in = np.random.uniform(
            -self.scale_i, self.scale_i, (self.size_r, size_i))
    
    def reset_reservoir_weight(self):
        # initialize the reservoir weight with uniform distribution
        num_elements = self.size_r * self.size_r
        W_res_flat = np.random.uniform(
            -1.0, 1.0, num_elements)
        # set sparsity of reservoir weight
        num_non_zero = int(num_elements * (1-self.sparsity))
        zero_indices = np.random.choice(num_elements, num_elements-num_non_zero, replace=False)
        W_res_flat[zero_indices] = 0.0
        self.W_res = W_res_flat.reshape(self.size_r, self.size_r)
		 # adjust spectral radius
        self.W_res = self.sr * self.W_res / np.max(np.abs(np.linalg.eigvals(self.W_res)))
        # store W_res as sparse matrix for efficiency
        self.W_res = sp.csr_matrix(self.W_res)
        
    def reset_state(self):
        # initialize the reservoir state with zeros
        self.state_r = np.zeros(self.size_r)
        # self.state_r = np.random.uniform(-1.0, 1.0, self.size_r)
        
    def forward(self, u=None):
        if self.W_in is None:
            self.reset_input_weight(len(u))

        if u is not None:
            
            # input to reservoir
            state_r = self.fn_nonlin(self.W_in @ u + self.W_res @ self.state_r)
            # leaking
            self.state_r = (1 - self.lr) * self.state_r + self.lr * state_r
            return self.state_r
        else:
            # no input to reservoir
            state_r = self.fn_nonlin(self.W_res @ self.state_r)
            self.state_r = (1 - self.lr) * self.state_r + self.lr * state_r
            return self.state_r

    def collect_states(self, U, do_reset_state=True):
        
        if self.W_in is None:
            _, n = U.shape
            # initialize input weight when receiving first input
            self.reset_input_weight(n)

        if do_reset_state:
            self.reset_state()
    
        states = []
        for u in U:
            state_r = self.forward(u)
            states.append(state_r)
        states = np.array(states)
        return states, self.state_r
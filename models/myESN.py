import numpy as np
np.random.seed(1234)

# count the number of zeros proportion in matrix
def count_zeros_proportion(matrix):
    num_zeros = np.count_nonzero(matrix == 0)
    total_elements = matrix.shape[0] * matrix.shape[1]
    return num_zeros / total_elements


# definition of original Echo State Network

# activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x)) 

def linear(x):
    return x

class original_ESN():
    def __init__(self, input_size, output_size, reservoir_size=100, spectral_radius=1.0,
                  sparsity=0.5, leaking_rate=1.0, input_scale=1.0, activation="tanh"):
        self.input_size = input_size
        self.output_size = output_size
        self.reservoir_size = reservoir_size
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.leaking_rate = leaking_rate
        self.input_scale = input_scale
        if activation == "tanh":
            # self.f = tanh
            self.f = np.tanh
        elif activation == "relu":
            self.f = relu
        elif activation == "sigmoid":
            self.f = sigmoid
        elif activation == "linear":
            self.f = linear
        
        self.reset_params()

    def reset_params(self):
        # generate the W_in, W_res and reset x_res, W_out
        # Readin weight
        # uniform distribution, [-scale, scale]
        self.W_in = np.random.uniform(-self.input_scale, self.input_scale, size=(self.reservoir_size, self.input_size))
        
        # Reservoir weight
        # uniform distribution, [-1.0, 1.0]
        # with predefined sparsity(density)
        num_elements = self.reservoir_size**2
        W_res = np.random.uniform(-1.0, 1.0, size=num_elements)
        zero_idx = np.random.permutation(num_elements)
        zero_idx = zero_idx[:int(num_elements * self.sparsity)]
        W_res[zero_idx] = 0
        W_res = W_res.reshape(-1, self.reservoir_size)
        W_res = W_res / np.max(np.abs(np.linalg.eigvals(W_res))) * self.spectral_radius
        self.W_res = W_res
        
		# Reservoir state
        self.reset_state()
		# Readout weight
        self.reset_readout()

    def reset_state(self):
        self.x_res = np.zeros(self.reservoir_size)

    def reset_readout(self):
        self.W_out = np.zeros((self.output_size, self.reservoir_size))

    def update(self, input_data):
        # update the reservoir state
        _x_res = self.f(self.W_in @ input_data + self.W_res @ self.x_res)
        self.x_res = (1 - self.leaking_rate) * self.x_res + self.leaking_rate * _x_res
        return self.x_res
    
    def get_output(self):
        return self.x_res @ self.W_out



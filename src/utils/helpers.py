import numpy as np

def compute_mse(x, x_hat):
    return np.mean((x - x_hat) ** 2, axis=1)

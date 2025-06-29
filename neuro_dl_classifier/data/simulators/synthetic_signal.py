import numpy as np
import os

def generate_synthetic_data(n_samples=1000, n_channels=32, signal_length=256, freq=10):
    X = np.random.randn(n_samples, n_channels, signal_length) * 0.1
    for i in range(n_samples):
        sine = np.sin(np.linspace(0, 2 * np.pi * freq, signal_length))
        X[i] += sine * np.random.rand(n_channels, 1)
    y = np.random.randint(0, 2, size=(n_samples,))
    return X, y

def save_synthetic_dataset(save_path="data/synthetic", **kwargs):
    os.makedirs(save_path, exist_ok=True)
    X, y = generate_synthetic_data(**kwargs)
    np.save(f"{save_path}/X.npy", X)
    np.save(f"{save_path}/y.npy", y)

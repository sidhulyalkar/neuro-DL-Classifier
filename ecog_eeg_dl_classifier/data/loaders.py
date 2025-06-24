import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

def load_signal_data(data_dir, batch_size=32):
    X = np.load(f"{data_dir}/X.npy")
    y = np.load(f"{data_dir}/y.npy")

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

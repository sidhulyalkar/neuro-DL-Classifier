import numpy as np
import time
import torch
from ecog_eeg_dl_classifier.models.cnn_baseline import EEGCNN

def stream_simulated_signal(model, freq=1):
    model.eval()
    while True:
        sample = np.sin(np.linspace(0, 2 * np.pi * 10, 256)) + np.random.randn(256) * 0.1
        input_tensor = torch.tensor(sample.reshape(1, 1, -1), dtype=torch.float32)
        with torch.no_grad():
            output = model(input_tensor)
            pred = torch.argmax(output, dim=1)
            print(f"Predicted class: {pred.item()}")
        time.sleep(freq)

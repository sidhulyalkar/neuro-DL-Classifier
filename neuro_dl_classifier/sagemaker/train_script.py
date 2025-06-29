if __name__ == "__main__":
    from ecog_eeg_dl_classifier.data.loaders import load_signal_data
    from ecog_eeg_dl_classifier.models.cnn_baseline import EEGCNN
    from ecog_eeg_dl_classifier.training.trainer import train_model
    import torch

    model = EEGCNN()
    dataloader = load_signal_data("/opt/ml/input/data/training")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_model(model, dataloader, device=device)

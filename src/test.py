import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

from src.network import Dataset, CNN
from src.utils import *

def test(device, params, model_path, logger):
    # Dataset e DataLoader
    test_dataset = Dataset(params['csv_path'], logger, train=False)
    test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=True)

    # Modello
    model = load_model(2, model_path, device)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing", unit="batch"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    logger.info(f"Test Accuracy: {acc:.4f}")
    logger.info("Classification Report:")
    logger.info("\n" + classification_report(all_labels, all_preds, digits=4))
    pass

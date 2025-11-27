import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
import csv

from src.network import Dataset, CNN
from src.utils import *


def train(device, params, logger, load, model_path=None):

    logger.info("Loading dataset...")
    train_datasets = []
    validation_datasets = []
    
    for csv_path in params['csv_path']:
        train_ds = Dataset(csv_path, logger, train=True)
        train_datasets.append(train_ds)
        sel_idx = [i for i, _ in enumerate(train_ds.samples)]
        val_idx = train_ds.get_validation_idx(sel_idx)
        validation_datasets.append(Dataset(csv_path, logger, train=False, sel_idx=val_idx))
    train_dataset = torch.utils.data.ConcatDataset(train_datasets)
    validation_dataset = torch.utils.data.ConcatDataset(validation_datasets)
    
    logger.info(f"Dataset loaded with {len(train_dataset)} samples.")
    logger.info(f"Validation dataset loaded with {len(validation_dataset)} samples.")

    # Dataset e DataLoader
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=params['batch_size'], shuffle=False)

    # Modello
    if load:
        logger.info("Loading existing model...")
        model = load_model(params['num_classes'], model_path, device)
    else:
        model = CNN(params['num_classes']).to(device)
    model.train()

    # Loss e ottimizzatore
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])

    # Per il plot
    loss_history = []
    acc_history = []

    # Creazione delle plot
    figL, axL = plt.subplots(figsize=(7, 4))
    figA, axA = plt.subplots(figsize=(7, 4))

    for epoch in range(params['epochs']):
        running_loss = 0.0
        correct = 0
        total = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{params['epochs']}", unit='batch')
        for inputs, labels in loop:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            bs = inputs.size(0)
            running_loss += loss.item() * bs
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += bs
            epoch_loss = running_loss / total
            epoch_acc = correct / total
            loop.set_postfix(loss=f"{epoch_loss:.4f}", acc=f"{epoch_acc:.4f}")
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        loss_history.append(epoch_loss)
        acc_history.append(epoch_acc)
        logger.info(f"Epoch {epoch+1}/{params['epochs']} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.4f}")

    # Creazione delle plot
    axL = update_plot(axL, loss_history, "blue", "Loss", "Loss")
    figL.tight_layout()

    axA = update_plot(axA, acc_history, "green", "Train Accuracy", "Accuracy")
    figA.tight_layout()

    # Salvataggio delle figure delle loss
    figL.savefig(join(PLOT_DIR, f"loss.png"), dpi=200, bbox_inches="tight")
    figA.savefig(join(PLOT_DIR, f"train_acc.png"), dpi=200, bbox_inches="tight")

    validation_acc_history = []
    validation_f1_history = []
    validation_precision_history = []
    validation_recall_history = []
    model.eval()
    with torch.no_grad():
        all_preds = []
        all_labels = []
        correct = 0
        total = 0
        for inputs, labels in tqdm(validation_loader, desc="Validation", unit="batch"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        validation_acc = correct / total
        validation_acc_history.append(validation_acc)
        logger.info("\n" + classification_report(all_labels, all_preds, digits=4))
        f1 = f1_score(all_labels, all_preds, average='weighted')
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        validation_f1_history.append(f1)
        validation_precision_history.append(precision)
        validation_recall_history.append(recall)
        logger.info(f"Validation Accuracy: {validation_acc:.4f}")
        logger.info(f"Validation F1-score: {f1:.4f}")
        logger.info(f"Validation Precision: {precision:.4f}")
        logger.info(f"Validation Recall: {recall:.4f}")

    # Salva le metriche di validazione in un CSV (append se esiste, crea se non esiste)
    import os
    metrics_path = join(PLOT_DIR, "validation_metrics.csv")
    file_exists = os.path.isfile(metrics_path)
    with open(metrics_path, "a" if file_exists else "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(["Epoch", "Accuracy", "F1-score", "Precision", "Recall"])
        for i in range(len(validation_acc_history)):
            writer.writerow([
                params['epochs'] if 'epochs' in params else i+1,
                validation_acc_history[i],
                validation_f1_history[i],
                validation_precision_history[i],
                validation_recall_history[i]
            ])
    logger.info(f"Validation metrics saved to {metrics_path}")

    # Fine epochs
    return model

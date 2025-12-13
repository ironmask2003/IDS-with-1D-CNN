import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm
import os
import joblib

from src.network_unsup import UNSWDataset, ConvAutoencoder
from src.utils import init_logger, set_device, MODEL_DIR


def load_model_and_threshold(model_path=None, threshold_path=None, input_len=None, device=None):
    if device is None:
        device = set_device()
    if model_path is None:
        model_path = os.path.join(MODEL_DIR, "conv_autoencoder.pth")
    if threshold_path is None:
        threshold_path = os.path.join(MODEL_DIR, "recon_threshold.json")

    # need input_len to instantiate model
    if input_len is None:
        raise ValueError("input_len must be provided to instantiate the model")

    # Load checkpoint first to infer latent_dim (avoid shape mismatch)
    ckpt = torch.load(model_path, map_location=device)
    latent_dim = None

    # Try to find bottleneck linear weight key in state dict to infer latent_dim
    for k, v in ckpt.items():
        if isinstance(k, str) and 'bottleneck' in k and k.endswith('.weight'):
            # weight shape typically [latent_dim, flatten_dim]
            latent_dim = v.shape[0]
            break

    if latent_dim is None:
        # Fallback: try some common keys or default to 128
        for k, v in ckpt.items():
            if isinstance(k, str) and 'bottleneck' in k:
                if v.dim() >= 2:
                    latent_dim = v.shape[0]
                    break

    if latent_dim is None:
        print("Warning: couldn't infer latent_dim from checkpoint; using default latent_dim=128")
        latent_dim = 128
    else:
        print(f"Inferred latent_dim={latent_dim} from checkpoint")

    model = ConvAutoencoder(input_len=input_len, latent_dim=latent_dim)
    model.load_state_dict(ckpt)
    model.to(device)
    model.eval()

    with open(threshold_path, "r") as f:
        jd = json.load(f)
    threshold = float(jd["threshold"])
    return model, threshold


def evaluate(eval_csv, model_path=None, threshold_path=None, scaler_path=None, batch_size=512, device=None, logger=None):
    if device is None:
        device = set_device()
    if logger is None:
        logger = init_logger("unsup-test", os.path.join("Logs", "unsup_test.log"))

    # Carica scaler se presente
    if scaler_path is None:
        scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    scaler = None
    if os.path.isfile(scaler_path):
        scaler = joblib.load(scaler_path)
        logger.info(f"Loaded scaler from {scaler_path}")
    else:
        logger.warning(f"No scaler found at {scaler_path}. Proceeding without scaling (not recommended).")

    # Carica dataset eval: qui vogliamo le etichette per calcolare metriche
    eval_ds = UNSWDataset(eval_csv, logger=logger, train=False, only_label_1=False, scaler=scaler)
    if eval_ds.y is None:
        raise RuntimeError("Evaluation CSV must contain a 'Label' or 'label' column for evaluation.")

    input_len = eval_ds.seq_len
    model, threshold = load_model_and_threshold(model_path=model_path, threshold_path=threshold_path, input_len=input_len, device=device)

    loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False)

    all_errors = []
    all_labels = []
    with torch.no_grad():
        for batch_x, batch_y in tqdm(loader, desc="Evaluating"):
            batch_x = batch_x.to(device)
            recon = model(batch_x)
            per_sample_mse = torch.mean((recon - batch_x) ** 2, dim=1)
            all_errors.append(per_sample_mse.cpu().numpy())
            all_labels.append(batch_y.numpy())

    all_errors = np.concatenate(all_errors, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Predizione: se error > threshold => anomalia => label_pred = 0 (assumiamo label 1 = inlier)
    preds = (all_errors <= threshold).astype(int)  # 1 = inlier, 0 = anomaly

    acc = accuracy_score(all_labels, preds)
    report = classification_report(all_labels, preds, digits=4)
    cm = confusion_matrix(all_labels, preds)

    logger.info(f"Threshold used: {threshold:.6f}")
    logger.info(f"Eval Accuracy: {acc:.4f}")
    logger.info("Classification Report:\n" + report)
    logger.info(f"Confusion Matrix:\n{cm.tolist()}")

    return {
        "accuracy": acc,
        "report": report,
        "confusion_matrix": cm,
        "threshold": threshold,
        "errors": all_errors,
        "labels": all_labels,
        "preds": preds
    }
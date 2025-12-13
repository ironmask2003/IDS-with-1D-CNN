import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import StandardScaler

from src.network_unsup import UNSWDataset, ConvAutoencoder
from src.utils import init_logger, set_device, PLOT_DIR, MODEL_DIR, LOG_DIR, update_plot


def train_unsupervised(train_csv,
                       model_out_path=None,
                       threshold_out_path=None,
                       scaler_out_path=None,
                       epochs=20,
                       batch_size=256,
                       lr=1e-3,
                       latent_dim=64,
                       only_label_1=True,
                       device=None,
                       logger=None):
    """
    Train a convolutional autoencoder on train_csv.
    - Fit a StandardScaler on the training features and save it.
    - only_label_1: if True and train csv contains Label column, keep only Label==1 rows for training.
    - Saves model and threshold (95th percentile on training reconstruction errors) to disk.
    """

    if device is None:
        device = set_device()
    if logger is None:
        logger = init_logger("unsup-train", os.path.join(LOG_DIR, "unsup_train.log"))

    os.makedirs(PLOT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    logger.info(f"Loading training dataset from {train_csv}")
    # Primo dataset "grezzo" per fit dello scaler (non normalizzato)
    tmp_ds = UNSWDataset(train_csv, logger=logger, train=True, only_label_1=only_label_1, scaler=None)
    # Fit StandardScaler sulle features del training
    logger.info("Fitting StandardScaler on training features...")
    scaler = StandardScaler()
    scaler.fit(tmp_ds.features)  # tmp_ds.features è numpy array float32
    if scaler_out_path is None:
        scaler_out_path = os.path.join(MODEL_DIR, "scaler.pkl")
    joblib.dump(scaler, scaler_out_path)
    logger.info(f"Saved scaler to {scaler_out_path}")

    # Ora dataset normalizzato (passiamo lo scaler)
    train_ds = UNSWDataset(train_csv, logger=logger, train=True, only_label_1=only_label_1, scaler=scaler)
    input_len = train_ds.seq_len
    logger.info(f"Training samples: {len(train_ds)}, feature length: {input_len}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0)

    model = ConvAutoencoder(input_len=input_len, latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss(reduction='mean')  # per-batch mean
    # considerare weight_decay se necessario: torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    loss_history = []

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        count = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch")
        for batch_x, _ in loop:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            recon = model(batch_x)
            loss_tensor = criterion(recon, batch_x)
            loss_tensor.backward()
            # gradient clipping per stabilità
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            running_loss += loss_tensor.item() * batch_x.size(0)
            count += batch_x.size(0)
            loop.set_postfix(loss=f"{running_loss / count:.6f}")

        epoch_loss = running_loss / max(1, count)
        loss_history.append(epoch_loss)
        logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.6f}")

    # Salva training loss plot
    fig, ax = plt.subplots(figsize=(7, 4))
    ax = update_plot(ax, loss_history, "blue", "Training Loss", "MSE")
    fig.tight_layout()
    loss_plot_path = os.path.join(PLOT_DIR, "unsup_loss.png")
    fig.savefig(loss_plot_path, dpi=200, bbox_inches="tight")
    logger.info(f"Saved loss plot to {loss_plot_path}")

    # Calcola errori di ricostruzione su tutto il training set (per sample)
    model.eval()
    train_errors = []
    with torch.no_grad():
        for batch_x, _ in tqdm(DataLoader(train_ds, batch_size=512), desc="Compute train recon errors"):
            batch_x = batch_x.to(device)
            recon = model(batch_x)
            per_sample_mse = torch.mean((recon - batch_x) ** 2, dim=1)  # (batch,)
            train_errors.append(per_sample_mse.cpu().numpy())
    train_errors = np.concatenate(train_errors, axis=0)
    # Soglia: 95° percentile (modificabile)
    threshold = float(np.percentile(train_errors, 95))

    # Salva modello e soglia
    if model_out_path is None:
        model_out_path = os.path.join(MODEL_DIR, "conv_autoencoder.pth")
    if threshold_out_path is None:
        threshold_out_path = os.path.join(MODEL_DIR, "recon_threshold.json")

    torch.save(model.state_dict(), model_out_path)
    with open(threshold_out_path, "w") as f:
        json.dump({"threshold": threshold}, f)

    logger.info(f"Saved model to {model_out_path}")
    logger.info(f"Saved reconstruction threshold ({threshold:.6f}) to {threshold_out_path}")

    return model_out_path, threshold_out_path, scaler_out_path
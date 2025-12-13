import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm
import pandas as pd


class UNSWDataset(Dataset):
    """
    Dataset robusto per i CSV UNSW:
    - legge il CSV (si assume header presente).
    - rimuove colonne testuali/not-numeric come srcip, dstip, proto, service, attack_cat.
    - converte le restanti colonne in numerico (coerce), fillna(0).
    - se fornito uno scaler (es. sklearn StandardScaler) applica scaler.transform alle features.
    - per training: se esiste la colonna 'Label' o 'label' filtra per label == 1 (se desired).
    - __getitem__ restituisce (features_tensor, label) se label_exists else (features_tensor, -1)
    """

    def __init__(self, csv_path, logger=None, train=True, only_label_1=False, scaler=None):
        self.csv_path = csv_path
        self.train = train
        self.only_label_1 = only_label_1
        self.scaler = scaler
        # Leggi CSV
        df = pd.read_csv(csv_path, header=0, encoding='latin1', on_bad_lines='skip', low_memory=False)

        # Normalizza nomi colonne (rimuove spazi strani)
        df.columns = [c.strip() for c in df.columns]

        # Individua colonna label se presente
        self.label_col = None
        if 'Label' in df.columns:
            self.label_col = 'Label'
        elif 'label' in df.columns:
            self.label_col = 'label'

        # Colonne tipicamente non numeriche (da escludere)
        non_numeric_candidates = ['srcip', 'dstip', 'proto', 'service', 'attack_cat', 'state']
        # Rimuovi colonne non numeriche se presenti
        cols_to_keep = []
        for c in df.columns:
            if c in non_numeric_candidates:
                continue
            cols_to_keep.append(c)

        # Se c'è label la teniamo separata
        if self.label_col and self.label_col not in cols_to_keep:
            cols_to_keep.append(self.label_col)

        df_sel = df[cols_to_keep].copy()

        # Converti tutte le colonne (tranne label) in numerico
        numeric_cols = [c for c in df_sel.columns if c != self.label_col]
        for c in numeric_cols:
            df_sel[c] = pd.to_numeric(df_sel[c], errors='coerce').fillna(0.0)

        # Se label esiste converti in int (riempi NaN con 0)
        if self.label_col:
            df_sel[self.label_col] = pd.to_numeric(df_sel[self.label_col], errors='coerce').fillna(0).astype(int)

        # Se training e only_label_1 True, filtra le righe con label==1 (se label disponibile)
        if train and self.only_label_1 and self.label_col:
            df_sel = df_sel[df_sel[self.label_col] == 1].reset_index(drop=True)

        # Se label presente, separa X e y
        if self.label_col:
            self.y = df_sel[self.label_col].values.astype(np.int64)
            Xdf = df_sel.drop(columns=[self.label_col])
        else:
            self.y = None
            Xdf = df_sel

        # Final features numpy (float32)
        X = Xdf.values.astype(np.float32)

        # Applica scaler se fornito (scaler deve avere metodo transform come sklearn)
        if self.scaler is not None:
            try:
                X = self.scaler.transform(X)
            except Exception as e:
                # nel caso lo scaler non accetti float32, ricastiamo
                X = self.scaler.transform(X.astype(np.float64)).astype(np.float32)

        self.features = X
        # numero di feature (lunghezza della "sequence")
        self.seq_len = X.shape[1]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        if self.y is None:
            return x, -1
        return x, int(self.y[idx])


class ConvAutoencoder(nn.Module):
    """
    Autoencoder convoluzionale 1D.
    Input shape: (batch, seq_len) -> trattato come (batch, 1, seq_len).
    Encoder: Conv1d + Pooling -> Flatten -> linear bottleneck
    Decoder: Linear -> Unflatten -> ConvTranspose1d -> output (batch, 1, seq_len) -> squeeze
    """

    def __init__(self, input_len, latent_dim=128):
        super().__init__()
        self.input_len = input_len

        # Encoder conv stack
        self.encoder_conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),  # L/2
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),  # L/4
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)   # L/8
        )

        # calcola dimensione flatten dinamicamente
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_len)
            enc_out = self.encoder_conv(dummy)
            self._enc_shape = enc_out.shape  # (batch, channels, seq_encoded)
            flatten_dim = enc_out.numel()

        self.flatten_dim = flatten_dim
        self.latent_dim = latent_dim

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_dim, latent_dim),
            nn.ReLU()
        )

        # Decoder linear then conv transpose
        self.decoder_lin = nn.Sequential(
            nn.Linear(latent_dim, flatten_dim),
            nn.ReLU()
        )

        # decoder conv transpose mirrors encoder_conv
        c, l = self._enc_shape[1], self._enc_shape[2]
        # We'll reshape to (batch, c, l) then apply transpose convs (mirror)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose1d(c, 32, kernel_size=2, stride=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 16, kernel_size=2, stride=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.ConvTranspose1d(16, 1, kernel_size=2, stride=2),
            # Output no activation (reconstruct raw values)
        )

    def encode(self, x):
        # x: (batch, seq_len)
        x = x.unsqueeze(1)  # (batch,1,L)
        z = self.encoder_conv(x)
        z_flat = z.view(z.size(0), -1)
        latent = self.bottleneck(z_flat)
        return latent

    def decode(self, latent):
        z_flat = self.decoder_lin(latent)
        z = z_flat.view(-1, self._enc_shape[1], self._enc_shape[2])
        recon = self.decoder_conv(z)  # (batch, 1, L_out)
        recon = recon.squeeze(1)  # (batch, L_out)

        # align reconstructed length to original input length (interpolate if needed)
        if recon.size(-1) != self.input_len:
            recon = F.interpolate(recon.unsqueeze(1), size=self.input_len, mode='linear', align_corners=False)
            recon = recon.squeeze(1)

        return recon

    def forward(self, x):
        latent = self.encode(x)
        recon = self.decode(latent)
        return recon
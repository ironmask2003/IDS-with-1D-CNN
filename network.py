import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm
import pandas as pd


########################################################
# Dataset PyTorch
########################################################
class Dataset(Dataset):
    def __init__(self, csv_path, max_len=1500):
        df = pd.read_csv(csv_path)
        # Carica la lista delle features dal file delle features
        numeric_features = []
        for elem in df.values[0]:
            numeric_features.append(df.columns[df.values[0].tolist().index(elem)]) if type(elem) in [int, float] else None
        print(numeric_features)
        # Escludi label e attack_cat
        feature_cols = [col for col in numeric_features if col not in ['label', 'attack_cat']]
        # Prendi solo le prime max_len features se necessario
        X = df[feature_cols].values.astype(np.float32)
        # Gestione label (può essere 'label' o 'Label')
        label_col = 'label' if 'label' in df.columns else 'Label'
        y = df[label_col].values.astype(np.int64)
        # Padding/troncamento
        X_padded = []
        for row in tqdm(X, desc="Padding sequences", unit="row", total=len(X)):
            if len(row) < max_len:
                padded = np.pad(row, (0, max_len - len(row)), 'constant')
            else:
                padded = row[:max_len]
            X_padded.append(padded)
        self.samples = list(zip(X_padded, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq, label = self.samples[idx]
        return torch.tensor(seq, dtype=torch.float32), label
    
class CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=8, stride=1, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(32, 64, kernel_size=8, stride=1, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(64, 128, kernel_size=8, stride=1, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(4),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * (1500 // (4*4*4)), 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch, 1, seq_len)
        x = self.net(x)
        return self.classifier(x)
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
    def __init__(self, csv_path, logger, max_len=2000, train=True, sel_idx=None):
        # Carica le info delle features
        features_info = pd.read_csv('./Dataset/NUSW-NB15_features.csv', encoding='latin1')
        # Prendi solo le colonne numeriche
        numeric_types = ['Integer', 'Float', 'Binary', 'integer', 'float', 'binary']
        numeric_features = features_info[features_info['Type '].str.strip().isin(numeric_types)]['Name'].tolist()
        # Aggiungi anche la colonna label (può essere 'label' o 'Label')
        label_col = 'label' if 'label' in numeric_features else 'Label'
        all_features = numeric_features + [label_col]
        # Carica il dataset senza header, usando i nomi delle colonne
        df = pd.read_csv(csv_path, header=None, names=features_info['Name'].tolist(), encoding='latin1', on_bad_lines='skip', low_memory=False)
        # Seleziona solo le colonne numeriche
        feature_cols = [col for col in numeric_features if col not in ['label', 'Label', 'attack_cat']]
        X = df[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(np.float32)
        y = pd.to_numeric(df[label_col], errors='coerce').fillna(0).astype(np.int64)
        self.total_len = len(X)
        
        if train:
            idx_1 = np.where(y == 1)[0]
            idx_0 = np.where(y == 0)[0]
            n_1 = int(len(idx_1) * 0.8)
            n_0 = int(len(idx_1) * 0.5)
            np.random.seed(39)
            sel_1 = np.random.choice(idx_1, n_1, replace=False)
            sel_0 = np.random.choice(idx_0, n_0, replace=False)
            sel_idx = np.concatenate([sel_1, sel_0])
            np.random.shuffle(sel_idx)
            X_selected = X[sel_idx]
            y_selected = y[sel_idx]
        else:
            # Se sel_idx è passato, usa solo quei dati (validation)
            if sel_idx is not None:
                X_selected = X[sel_idx]
                y_selected = y[sel_idx]
            else:
                X_selected = X
                y_selected = y

        # Padding/troncamento
        X_padded = []
        for row in tqdm(X_selected, desc="Padding sequences", unit="row", total=len(X_selected)):
            if len(row) < max_len:
                padded = np.pad(row, (0, max_len - len(row)), 'constant')
            else:
                padded = row[:max_len]
            X_padded.append(padded)
        self.samples = list(zip(X_padded, y_selected))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq, label = self.samples[idx]
        return torch.tensor(seq, dtype=torch.float32), label
    
    def get_validation_idx(self, sel_idx):
        all_idx = set(range(self.total_len))
        val_idx = np.array(list(all_idx - set(sel_idx)))
        return val_idx
    
class CNN(nn.Module):
    def __init__(self, num_classes, input_len=2000):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=8, stride=1, padding=4),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(4)
        )
        # Calcola la dimensione finale dinamicamente
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_len)
            out = self.net(dummy)
            flatten_dim = out.numel()

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch, 1, seq_len)
        x = self.net(x)
        return self.classifier(x)
import torch
import numpy as np
import joblib
import logging
import os
from os.path import join, abspath, dirname, pardir

# Directory di base
BASE_DIR = abspath(join(dirname(__file__), pardir))

PLOT_DIR = join(BASE_DIR, "Plots")
MODEL_DIR = join(BASE_DIR, "Model")

LOG_DIR = join(BASE_DIR, "Logs")

LOG_FORMAT = "%(asctime)s %(name)-12s %(levelname)-8s %(message)s"

def init_dirs():
    os.makedirs(PLOT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

def init_logger(name, log_dir=None):
    """
    Inizializza il logger per il modulo specificato.

    Args:
        - name : nome del modulo
        - log_dir : percorso del file di log (opzionale)
    """

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    # create formatter
    formatter = logging.Formatter(LOG_FORMAT)
    # add formatter to ch
    ch.setFormatter(formatter)
    # add ch to logger
    logger.addHandler(ch)

    if log_dir is not None:
        ch2 = logging.StreamHandler(open(log_dir, 'w'))
        ch2.setFormatter(formatter)
        logger.addHandler(ch2)
    return logger


def set_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    return device

def params(logger, test, csv_path='./Dataset/UNSW-NB15_1.csv', batch_size=64, epochs=10, lr=2e-3, num_classes=2):

    csv_paths = ["./Dataset/UNSW-NB15_1.csv", "./Dataset/UNSW-NB15_2.csv", "./Dataset/UNSW-NB15_3.csv", "./Dataset/UNSW-NB15_4.csv"]
    train_path = "./Dataset/UNSW_top5_train.csv"
    eval_path = "./Dataset/UNSW_top5_eval.csv"
    logger.info(f"Using default parameters: csv_path={csv_paths}, batch_size={batch_size}, epochs={epochs}, lr={lr}, num_classes={num_classes}")
    return {
        'train_path': train_path,
        'eval_path': eval_path,
        'batch_size': batch_size,
        'epochs': epochs,
        'lr': lr,
        'num_classes': num_classes
    }

def save_model(model, model_path):
    if model_path is None:
        model_path = join(MODEL_DIR, "cnn_model.pth")
    torch.save(model.state_dict(), model_path)

def save_rf_model(model, model_path):
    if model_path is None:
        model_path = join(MODEL_DIR, "rf_model.joblib")
    joblib.dump(model, model_path)

def update_plot(ax, values, color, title, label_title):
    """
    Funzione per aggiornare il grafico passato in input.

    Args:
        - ax : asse del grafico da aggiornare
        - values : valori con cui aggiornare il grafico
        - color : colore da utilizzare per il grafico
        - title : titolo del grafico
    """

    ax.cla()
    x = np.arange(1, len(values) + 1)
    y = np.array(values, dtype=float)
    if len(x) > 0:
        ax.plot(x, y, color=f"tab:{color}", linewidth=1, label = label_title)
    _style(ax, title)
    ax.legend(loc="best")

    return ax

def _style(ax, title: str):
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, color=(0.9, 0.9, 0.9), linewidth=0.8)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
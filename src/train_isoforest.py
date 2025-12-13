import os
import json
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

from src.network_unsup import UNSWDataset
from src.utils import init_logger, PLOT_DIR, MODEL_DIR, LOG_DIR


def train_isolation_forest(
    train_csv,
    model_out_path=None,
    threshold_out_path=None,
    scaler_out_path=None,
    contamination=0.01,
    n_estimators=300,
    max_samples="auto",
    max_features=1.0,
    bootstrap=False,
    random_state=42,
    logger=None,
):
    """
    Allena un Isolation Forest usando le feature preprocessate da `UNSWDataset`.

    - Fit di uno StandardScaler sulle feature del training e salvataggio.
    - Allena un `sklearn.ensemble.IsolationForest` su tutte le feature normalizzate.
    - Calcola anomaly scores sul training e imposta la soglia al percentile (1 - contamination).
    - Salva modello, scaler e soglia su disco.

    Ritorna: (model_out_path, threshold_out_path, scaler_out_path)
    """

    if logger is None:
        logger = init_logger("isoforest-train", os.path.join(LOG_DIR, "isoforest_train.log"))

    os.makedirs(PLOT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    logger.info(f"Loading training dataset from {train_csv}")

    # 1) Dataset grezzo per fit dello scaler (train CSV di data_extraction NON contiene Label)
    tmp_ds = UNSWDataset(
        train_csv,
        logger=logger,
        train=True,
        only_label_1=False,
        scaler=None,
    )

    logger.info("Fitting StandardScaler on training features for IsolationForest...")
    # Se disponibile la colonna Label, filtra i normali (Label==0)
    features_for_scaler = tmp_ds.features
    if getattr(tmp_ds, "y", None) is not None:
        mask_norm = (tmp_ds.y == 0)
        features_for_scaler = features_for_scaler[mask_norm]
        logger.info(f"Scaler fit on normals: {features_for_scaler.shape[0]} samples")

    scaler = StandardScaler()
    scaler.fit(features_for_scaler)

    if scaler_out_path is None:
        scaler_out_path = os.path.join(MODEL_DIR, "scaler_isoforest.pkl")
    joblib.dump(scaler, scaler_out_path)
    logger.info(f"Saved scaler to {scaler_out_path}")

    # 2) Dataset normalizzato
    train_ds = UNSWDataset(
        train_csv,
        logger=logger,
        train=True,
        only_label_1=False,
        scaler=scaler,
    )
    X = train_ds.features
    logger.info(f"Training samples (all): {len(train_ds)}, feature length: {X.shape[1]}")

    # Se disponibile la colonna Label, usa solo i normali per il fit dell'IF
    if getattr(train_ds, "y", None) is not None:
        mask_norm = (train_ds.y == 0)
        X_fit = X[mask_norm]
        logger.info(f"IsolationForest fit on normals: {X_fit.shape[0]} samples")
    else:
        X_fit = X

    # 3) Modello IsolationForest
    model = IsolationForest(
        n_estimators=n_estimators,
        max_samples=max_samples,
        contamination=contamination,
        max_features=max_features,
        bootstrap=bootstrap,
        random_state=random_state,
        n_jobs=-1,
        verbose=0,
    )

    logger.info(
        "Training IsolationForest with params: "
        f"n_estimators={n_estimators}, max_samples={max_samples}, "
        f"contamination={contamination}, max_features={max_features}, bootstrap={bootstrap}"
    )
    model.fit(X_fit)

    # 4) Calcolo degli anomaly scores sul training
    train_scores = -model.score_samples(X)  # più alto = più anomalo

    # Soglia al percentile (1 - contamination). Se contamination è None, usa 95° percentile.
    if contamination is None:
        perc = 95.0
    else:
        perc = 100.0 * (1.0 - float(contamination))
    threshold = float(np.percentile(train_scores, perc))

    # 5) Salvataggi
    if model_out_path is None:
        model_out_path = os.path.join(MODEL_DIR, "isolation_forest.pkl")
    if threshold_out_path is None:
        threshold_out_path = os.path.join(MODEL_DIR, "isoforest_threshold.json")

    joblib.dump(model, model_out_path)
    with open(threshold_out_path, "w") as f:
        json.dump(
            {
                "threshold": threshold,
                "percentile": perc,
                "contamination": contamination,
                "score_type": "-score_samples (higher is more anomalous)",
            },
            f,
        )

    logger.info(f"Saved IsolationForest model to {model_out_path}")
    logger.info(f"Saved anomaly threshold ({threshold:.6f}) to {threshold_out_path}")

    # 6) Plot della distribuzione degli score con soglia
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(train_scores, bins=60, color="steelblue", alpha=0.85)
    ax.axvline(threshold, color="red", linestyle="--", label=f"threshold (p={perc:.1f})")
    ax.set_title("IsolationForest - Train anomaly scores")
    ax.set_xlabel("anomaly score (-score_samples)")
    ax.set_ylabel("count")
    ax.legend()
    fig.tight_layout()
    plot_path = os.path.join(PLOT_DIR, "isoforest_scores.png")
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    logger.info(f"Saved score histogram to {plot_path}")

    return model_out_path, threshold_out_path, scaler_out_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train IsolationForest on UNSW dataset")
    parser.add_argument("train_csv", type=str, help="Path to training CSV")
    parser.add_argument("--contamination", type=float, default=0.01, help="Expected outlier fraction")
    parser.add_argument("--n_estimators", type=int, default=300, help="Number of trees")
    parser.add_argument("--max_samples", default="auto", help="max_samples for each tree")
    parser.add_argument("--max_features", type=float, default=1.0, help="max_features for each tree")
    parser.add_argument("--bootstrap", action="store_true", help="Use bootstrap samples")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed")
    parser.add_argument("--model_out", type=str, default=None, help="Output path for model .pkl")
    parser.add_argument("--threshold_out", type=str, default=None, help="Output path for threshold .json")
    parser.add_argument("--scaler_out", type=str, default=None, help="Output path for scaler .pkl")

    args = parser.parse_args()

    train_isolation_forest(
        train_csv=args.train_csv,
        model_out_path=args.model_out,
        threshold_out_path=args.threshold_out,
        scaler_out_path=args.scaler_out,
        contamination=args.contamination,
        n_estimators=args.n_estimators,
        max_samples=args.max_samples,
        max_features=args.max_features,
        bootstrap=args.bootstrap,
        random_state=args.random_state,
    )

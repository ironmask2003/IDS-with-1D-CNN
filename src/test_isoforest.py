import os
import json
import numpy as np
import joblib

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.network_unsup import UNSWDataset
from src.utils import init_logger, MODEL_DIR, LOG_DIR

def evaluate_isolation_forest(
    eval_csv,
    model_path=None,
    threshold_path=None,
    scaler_path=None,
    logger=None,
):
    """
    Valuta un Isolation Forest su un CSV di evaluation prodotto da data_extraction.
    Il file eval DEVE contenere le feature e la colonna Label/label.
    """

    if logger is None:
        logger = init_logger("isoforest-eval", os.path.join(LOG_DIR, "isoforest_eval.log"))

    if model_path is None:
        model_path = os.path.join(MODEL_DIR, "isolation_forest.pkl")
    if threshold_path is None:
        threshold_path = os.path.join(MODEL_DIR, "isoforest_threshold.json")
    if scaler_path is None:
        scaler_path = os.path.join(MODEL_DIR, "scaler_isoforest.pkl")

    # Carica scaler e modello
    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)
    with open(threshold_path, "r") as f:
        jd = json.load(f)
    threshold = float(jd["threshold"])

    # Dataset eval (con etichette)
    eval_ds = UNSWDataset(eval_csv, logger=logger, train=False, only_label_1=False, scaler=scaler)
    if eval_ds.y is None:
        raise RuntimeError("Evaluation CSV must contain a 'Label' or 'label' column for evaluation.")

    X = eval_ds.features
    y_true = eval_ds.y

    # Score: -score_samples -> più alto = più anomalo
    scores = -model.score_samples(X)
    y_pred = (scores <= threshold).astype(int)  # 1=inlier(benigno), 0=anomalia(attacco)

    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=4)
    cm = confusion_matrix(y_true, y_pred)

    logger.info(f"Threshold used: {threshold:.6f}")
    logger.info(f"Eval Accuracy: {acc:.4f}")
    logger.info("Classification Report:\n" + report)
    logger.info(f"Confusion Matrix:\n{cm.tolist()}")

    return {
        "accuracy": acc,
        "report": report,
        "confusion_matrix": cm,
        "threshold": threshold,
        "scores": scores,
        "labels": y_true,
        "preds": y_pred,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate IsolationForest on UNSW eval dataset")
    parser.add_argument("eval_csv", type=str, help="Path to eval CSV")
    parser.add_argument("--model", type=str, default=None, help="Path to model .pkl")
    parser.add_argument("--threshold", type=str, default=None, help="Path to threshold .json")
    parser.add_argument("--scaler", type=str, default=None, help="Path to scaler .pkl")

    args = parser.parse_args()

    evaluate_isolation_forest(
        eval_csv=args.eval_csv,
        model_path=args.model,
        threshold_path=args.threshold,
        scaler_path=args.scaler,
    )

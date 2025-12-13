#!/usr/bin/env python3
"""
Script di utilità per training e testing unsupervised (AE o Isolation Forest):
Esempi:
    # Train Autoencoder
    python run_unsup.py --mode train --type ae --train_csv Dataset/UNSW_top5_train.csv --epochs 30

    # Test Autoencoder (usa model/threshold/scaler di default in Model/)
    python run_unsup.py --mode test --type ae --eval_csv Dataset/UNSW_top5_eval.csv

    # Train Isolation Forest
    python run_unsup.py --mode train --type isoforest --train_csv Dataset/UNSW_top5_train.csv --contamination 0.01

    # Test Isolation Forest
    python run_unsup.py --mode test --type isoforest --eval_csv Dataset/UNSW_top5_eval.csv
"""

import argparse
import os
from src.train_unsup import train_unsupervised
from src.test_unsup import evaluate
from src.train_isoforest import train_isolation_forest
from src.test_isoforest import evaluate_isolation_forest
from src.utils import init_logger, set_device, init_dirs

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "test"], required=True)
    parser.add_argument("--type", choices=["ae", "isoforest"], default="ae",
                        help="Seleziona il metodo: 'ae' (autoencoder) o 'isoforest'")
    parser.add_argument("--train_csv", default="Dataset/UNSW_top5_train.csv")
    parser.add_argument("--eval_csv", default="Dataset/UNSW_top5_eval.csv")
    parser.add_argument("--model_out", default=None)
    parser.add_argument("--threshold_out", default=None)
    parser.add_argument("--model_in", default=None)
    parser.add_argument("--threshold_in", default=None)
    # Parametri Autoencoder
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--latent_dim", type=int, default=64)
    # Parametri Isolation Forest
    parser.add_argument("--contamination", type=float, default=0.01)
    parser.add_argument("--n_estimators", type=int, default=300)
    parser.add_argument("--max_samples", default="auto")
    parser.add_argument("--max_features", type=float, default=1.0)
    parser.add_argument("--bootstrap", action="store_true")
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--build_dataset", action="store_true",
                        help="If set, build the dataset before training/testing.")
    return parser.parse_args()

def main():
    args = parse_args()
    init_dirs()
    logger = init_logger(f"run_{args.type}", os.path.join("Logs", f"{args.type}_run.log"))
    device = set_device() if args.device is None else args.device

    if args.build_dataset:
        from src.data_extraction import build_datasets
        logger.info("Building datasets...")
        build_datasets()
        logger.info("Datasets built.")

    if args.mode == "train":
        if args.type == "ae":
            # train_unsupervised ora ritorna (model_out_path, threshold_out_path, scaler_out_path)
            model_out, threshold_out, scaler_out = train_unsupervised(
                train_csv=args.train_csv,
                model_out_path=args.model_out,
                threshold_out_path=args.threshold_out,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                latent_dim=args.latent_dim,
                device=device,
                logger=logger,
            )
        else:
            # Isolation Forest training
            model_out, threshold_out, scaler_out = train_isolation_forest(
                train_csv=args.train_csv,
                model_out_path=args.model_out,
                threshold_out_path=args.threshold_out,
                contamination=args.contamination,
                n_estimators=args.n_estimators,
                max_samples=args.max_samples,
                max_features=args.max_features,
                bootstrap=args.bootstrap,
                random_state=args.random_state,
                logger=logger,
            )
        logger.info(f"Training finished. model={model_out}, threshold={threshold_out}, scaler={scaler_out}")
    else:
        # Test mode
        if args.type == "ae":
            res = evaluate(
                eval_csv=args.eval_csv,
                model_path=args.model_in,
                threshold_path=args.threshold_in,
                batch_size=args.batch_size,
                device=device,
                logger=logger,
            )
        else:
            res = evaluate_isolation_forest(
                eval_csv=args.eval_csv,
                model_path=args.model_in,
                threshold_path=args.threshold_in,
                scaler_path=None,
                logger=logger,
            )
        logger.info("Done. Results in dictionary returned from evaluation.")

if __name__ == "__main__":
    main()
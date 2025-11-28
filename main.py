import argparse

from src.train import *
from src.test import *
from src.utils import *

def parse_args():

    parser = argparse.ArgumentParser(description="Train a CNN model for text editing tasks.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training.")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer.")
    parser.add_argument("--model_path", type=str, default=None, help="Path to save the trained model.")
    parser.add_argument("--csv_path", type=str, default='./Dataset/UNSW-NB15_1.csv', help="Path to the training CSV dataset.")
    parser.add_argument("--test", action='store_true', help="If True, run in test mode.")
    parser.add_argument("--load", action='store_true', help="If True, load existing model for training continuation.")
    parser.add_argument("--model_type", type=str, default="cnn", choices=["cnn", "rf"], help="Type of model to train: 'cnn' or 'rf' (Random Forest).")

    return parser.parse_args()

def main():
    args = parse_args()

    # Set device
    device = set_device()

    if args.test:
        # Init logger
        logger = init_logger("Testing model", f"{LOG_DIR}/testing.log")
        test_params = params(logger, test=True)
        for idx, csv_path in enumerate(test_params['csv_path']):
            logger.info(f"Testing on dataset {idx+1} with csv_path: {csv_path}")
            test_params_single = {
                'csv_path': csv_path,
                'batch_size': test_params['batch_size'],
                'num_classes': test_params['num_classes']
            }
            test(device, test_params_single, args.model_path, logger)
        return
    else:

        if args.model_type == "rf":
            # Init logger
            logger = init_logger("Training Random Forest model", f"{LOG_DIR}/rf_training.log")
            training_params = params(logger, test=False, csv_path=args.csv_path, batch_size=args.batch_size, epochs=args.epochs, lr=args.learning_rate, num_classes=2)

            rf_model = train_randomf(training_params, logger, args.load)
            save_rf_model(rf_model, args.model_path)
            return
        # Init logger
        logger = init_logger("Training model", f"{LOG_DIR}/training.log")
        training_params = params(logger, test=False, csv_path=args.csv_path, batch_size=args.batch_size, epochs=args.epochs, lr=args.learning_rate, num_classes=2)

        trained_model = train_cnn(device, training_params, logger, args.load, args.model_path)
        save_model(trained_model, args.model_path)

if __name__ == "__main__":
    main()
import argparse
import os
import json

import joblib
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from utils import find_single_csv, load_X_y

SEED = 42

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training_folder",
        type=str,
        required=True,
        help="Folder containing training CSV (train.csv)",
    )
    parser.add_argument(
        "--testing_folder",
        type=str,
        required=True,
        help="Folder containing testing CSV (test.csv)",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        required=True,
        help="Output folder (inside AzureML 'outputs')",
    )
    parser.add_argument(
        "--target_column",
        type=str,
        default="diabetes", 
        help="Name of target column in the dataset",
    )
    parser.add_argument(
        "--n_estimators",
        type=int,
        default=200,
        help="Number of trees for LightGBM",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.05,
        help="Learning rate for LightGBM",
    )
    args = parser.parse_args()

    print(" ".join(f"{k}={v}" for k, v in vars(args).items()))
    
    train_csv = find_single_csv(args.training_folder)
    test_csv = find_single_csv(args.testing_folder)

    print(f"Training CSV: {train_csv}")
    print(f"Testing CSV: {test_csv}")

    X_train, y_train = load_X_y(train_csv, args.target_column)
    X_test, y_test = load_X_y(test_csv, args.target_column)

    print("Train shape before encoding:", X_train.shape, y_train.shape)
    print("Test shape before encoding:", X_test.shape, y_test.shape)

    # --- Encode categorical (object) columns ---
    cat_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
    print("Categorical columns:", cat_cols)

    if cat_cols:
        import pandas as pd

        # combine train and test to ensure same one-hot columns
        combined = pd.concat(
            [X_train, X_test],
            axis=0,
            keys=["train", "test"]
        )

        combined_encoded = pd.get_dummies(
            combined,
            columns=cat_cols,
            drop_first=True 
        )

        # split back into train and test
        X_train = combined_encoded.xs("train")
        X_test = combined_encoded.xs("test")

    print("Train shape after encoding:", X_train.shape, y_train.shape)
    print("Test shape after encoding:", X_test.shape, y_test.shape)

    # Model
    model = LGBMClassifier(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        random_state=SEED,
    )


    model.fit(X_train, y_train)

    # Evaluate
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    auc = roc_auc_score(y_test, y_proba)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("Classification report:")
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:")
    print(cm)

    # save metrics
    metrics_dir = os.path.join(args.output_folder, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    metrics = {
        "auc": float(auc),
        "accuracy": float(acc),
        "f1_score": float(f1),
    }
    with open(os.path.join(metrics_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f)
    print("Saved metrics to:", os.path.join(metrics_dir, "metrics.json"))

    # save model
    model_dir = os.path.join(args.output_folder, "model")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.pkl")
    joblib.dump({"model": model, "feature_names": list(X_train.columns)}, model_path)
    print("Saved model to:", model_path)

    print("DONE TRAINING")


if __name__ == "__main__":
    main()
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

    # Make sure output folder exists
    os.makedirs(args.output_folder, exist_ok=True)

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
        combined = pd.concat([X_train, X_test], axis=0, keys=["train", "test"])

        combined_encoded = pd.get_dummies(
            combined,
            columns=cat_cols,
            drop_first=True,
        )

        # split back into train and test
        X_train = combined_encoded.xs("train")
        X_test = combined_encoded.xs("test")

    print("Train shape after encoding:", X_train.shape, y_train.shape)
    print("Test shape after encoding:", X_test.shape, y_test.shape)

    feature_names = list(X_train.columns)

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

    # ---- Save artifacts in a "register-friendly" layout ----
    # 1) metrics.json at root (easy to find)
    metrics = {
        "auc": float(auc),
        "accuracy": float(acc),
        "f1_score": float(f1),
        "n_estimators": int(args.n_estimators),
        "learning_rate": float(args.learning_rate),
        "target_column": args.target_column,
        "n_features": int(len(feature_names)),
    }
    metrics_path = os.path.join(args.output_folder, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print("Saved metrics to:", metrics_path)

    # 2) feature_names.json at root
    feature_names_path = os.path.join(args.output_folder, "feature_names.json")
    with open(feature_names_path, "w") as f:
        json.dump(feature_names, f, indent=2)
    print("Saved feature names to:", feature_names_path)

    # 3) model.pkl at root (this is what you’ll load in inference too)
    model_path = os.path.join(args.output_folder, "model.pkl")
    joblib.dump({...}, model_path)
    print("Saved model to:", model_path)

    # 4) optional: model_info.json (helpful when debugging downloads)
    model_info = {
        "artifact": "model.pkl",
        "framework": "lightgbm",
        "task": "binary_classification",
        "seed": SEED,
    }
    model_info_path = os.path.join(args.output_folder, "model_info.json")
    with open(model_info_path, "w") as f:
        json.dump(model_info, f, indent=2)
    print("Saved model info to:", model_info_path)

    print("DONE TRAINING")


if __name__ == "__main__":
    main()

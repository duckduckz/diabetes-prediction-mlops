import os
import argparse
import logging

import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    
    SEED = 42
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        help="Path to cleaned diabetes data",
        required=True,
    )
    parser.add_argument(
        "--training_data_output",
        type=str,
        help="Path to folder where training CSV will be saved",
        required=True,       
    )
    parser.add_argument(
        "--testing_data_output",
        type=str,
        help="Path to folder where testing CSV will be saved",
        required=True,
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Fraction of data to use as test set (e.g. 0.2 for 20%)",
    )
    parser.add_argument(
        "--target_column",
        type=str,
        default="diabetes", 
        help="Name of the target column for stratified splitting",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logging.info(" ".join(f"{k}={v}" for k, v in vars(args).items()))

    input_path = args.data
    train_out_dir = args.training_data_output
    test_out_dir = args.testing_data_output
    test_size = args.test_size
    target_column = args.target_column

    os.makedirs(train_out_dir, exist_ok=True)
    os.makedirs(test_out_dir, exist_ok=True)
    
    # Figure out the actual CSV path (folder or file)
    if os.path.isdir(input_path):
        # Expect exactly one CSV produced by dataprep.py
        csv_files = [f for f in os.listdir(input_path) if f.endswith(".csv")]
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in folder: {input_path}")
        csv_path = os.path.join(input_path, csv_files[0])
    else:
        csv_path = input_path

    logging.info(f"Reading cleaned data from: {csv_path}")
    df = pd.read_csv(csv_path)

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in dataset. "
            f"Available columns: {df.columns.tolist()}"
        )

    y = df[target_column]
    X = df.drop(columns=[target_column])

    logging.info(f"Total rows before split: {len(df)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=SEED,
        stratify=y,
    )

    train_df = X_train.copy()
    train_df[target_column] = y_train

    test_df = X_test.copy()
    test_df[target_column] = y_test

    train_path = os.path.join(train_out_dir, "train.csv")
    test_path = os.path.join(test_out_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    logging.info(f"Train set rows: {len(train_df)}, saved to: {train_path}")
    logging.info(f"Test set rows:  {len(test_df)}, saved to: {test_path}")


if __name__ == "__main__":
    main()
import os
import argparse
import logging
import pandas as pd
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="path to input CSV file")
    parser.add_argument("--output_data", type=str, help="path to output folder")
    parser.add_argument(
        "--target_column",
        type=str,
        default="diabetes", 
        help="Name of the target column in the dataset",
    )
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    logging.info(" ".join(f"{k}={v}" for k, v in vars(args).items()))
    
    input_path = args.data
    output_dir = args.output_data
    target_column = args.target_column
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    logging.info(f"Reading input data from: {input_path}")
    df = pd.read_csv(input_path)

    logging.info(f"Columns in dataset: {df.columns.tolist()}")
    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in dataset. "
            f"Available columns: {df.columns.tolist()}"
        )
    
    # drop rows where target is missing
    before_rows = len(df)
    df = df.dropna(subset=[target_column])
    after_rows = len(df)
    logging.info(f"Dropped {before_rows - after_rows} rows with missing target.")
    
    # fill numeric nan with colm medians
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    logging.info(f"Numeric columns: {numeric_cols}")

    for col in numeric_cols:
        if df[col].isna().any():
            median_value = df[col].median()
            df[col] = df[col].fillna(median_value)
            logging.info(f"Filled NaNs in '{col}' with median={median_value}")

    # save cleaned dataset
    output_file = os.path.join(output_dir, "diabetes_clean.csv")
    df.to_csv(output_file, index=False)
    logging.info(f"Saved cleaned dataset to: {output_file}")


if __name__ == "__main__":
    main()
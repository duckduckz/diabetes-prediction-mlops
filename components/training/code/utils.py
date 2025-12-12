import os
import pandas as pd

def find_single_csv(folder: str) -> str:
    
    files = [f for f in os.listdir(folder) if f.endswith(".csv")]
    
    if not files:
        raise FileNotFoundError(f"No CSV files found in folder: {folder}")
    if len(files) > 1:
        print(
            f"Warning: multiple CSV files found in {folder}, "
            f"using the first one: {files[0]}"
        )
    return os.path.join(folder, files[0])

def load_X_y(csv_path: str, target_column: str):
    df = pd.read_csv(csv_path)
    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in dataset. "
            f"Available columns: {df.columns.tolist()}"
        )
    y = df[target_column]
    X = df.drop(columns=[target_column])
    return X, y
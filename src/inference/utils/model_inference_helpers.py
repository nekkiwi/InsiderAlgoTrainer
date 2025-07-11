# In src/inference/utils/model_inference_helpers.py

import os
import pandas as pd

def load_inference_data(file_path: str) -> pd.DataFrame:
    """
    Loads data for inference from a given Excel file path.
    Converts 'Filing Date' to datetime format.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Inference data file not found at: {file_path}")
        
    try:
        df = pd.read_excel(file_path)
        # Ensure date formatting is handled correctly after loading
        df['Filing Date'] = pd.to_datetime(df['Filing Date'], dayfirst=True, errors='coerce')
        print(f"- Successfully loaded inference data from {os.path.basename(file_path)} ({len(df)} rows).")
        return df
    except Exception as e:
        print(f"- Failed to load inference data from {file_path}: {e}")
        return pd.DataFrame()


import os
import pandas as pd

def load_feature_data(file_path):
    """Load the feature data from an Excel file and extract Ticker and Filing Date."""
    data_dir = os.path.join(os.path.dirname(__file__), '../../../data')
    file_path = os.path.join(data_dir, file_path)
    if os.path.exists(file_path):
        try:
            data = pd.read_excel(file_path)
            print(f"- Sheet successfully loaded from {file_path}.")
            return data
        except Exception as e:
            print(f"- Failed to load sheet from {file_path}: {e}")
            return None
    else:
        print(f"- File '{file_path}' does not exist.")
        return None
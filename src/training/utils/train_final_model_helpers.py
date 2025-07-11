# In src/training/utils/train_final_model_helpers.py

import os
import pandas as pd
import joblib

def load_data(features_file: str, targets_file: str):
    """Loads and prepares features and targets from Excel files for final training."""
    features_df = pd.read_excel(features_file)
    targets_dict_raw = pd.read_excel(targets_file, sheet_name=None)
    
    features_df['Filing Date'] = pd.to_datetime(features_df['Filing Date'], dayfirst=True, errors='coerce')
    
    targets_dict = {}
    for sheet_name, df in targets_dict_raw.items():
        df['Filing Date'] = pd.to_datetime(df['Filing Date'], dayfirst=True, errors='coerce')
        targets_dict[sheet_name] = df
        
    return features_df, targets_dict

def load_selected_features(sel_file: str, category: str, strategy_name: str, top_n: int) -> list:
    """Loads the top N features for a given strategy from the feature selection file."""
    sheet_name = f"{category.title()}_Features"
    try:
        df = pd.read_excel(sel_file, sheet_name=sheet_name)
        if strategy_name not in df.columns:
            raise ValueError(f"Strategy '{strategy_name}' not found in sheet '{sheet_name}'.")
            
        scores = df[['Feature', strategy_name]].copy().dropna(subset=[strategy_name])
        top_features = scores[scores[strategy_name] > 0].sort_values(by=strategy_name, ascending=False)
        
        print(f"Loaded top {top_n} features for strategy '{strategy_name}'.")
        return top_features['Feature'].head(top_n).tolist()
        
    except (FileNotFoundError, ValueError) as e:
        print(f"[ERROR] Could not load features for '{strategy_name}': {e}")
        return []

def save_final_models(classifier, regressor, base_models_dir: str, model_type: str, category: str, 
                      timepoint: str, threshold_pct: int, top_n: int, seed: int):
    """Saves the final trained classifier and regressor models to disk."""
    # Create a unique, descriptive directory for this specific strategy
    strategy_dir_name = f"{model_type}_{category}_{timepoint}_{threshold_pct}pct_top{top_n}"
    save_dir = os.path.join(base_models_dir, strategy_dir_name)
    os.makedirs(save_dir, exist_ok=True)

    # Define descriptive filenames for each model including the seed
    clf_filename = f"final_clf_seed{seed}.joblib"
    reg_filename = f"final_reg_seed{seed}.joblib"
    
    clf_path = os.path.join(save_dir, clf_filename)
    reg_path = os.path.join(save_dir, reg_filename)
    
    # Save the models using joblib
    joblib.dump(classifier, clf_path)
    if regressor:
        joblib.dump(regressor, reg_path)
        
    print(f"  - Saved models for seed {seed} to '{os.path.basename(save_dir)}'.")


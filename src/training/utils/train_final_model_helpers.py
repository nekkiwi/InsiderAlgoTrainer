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

def save_final_models(classifier, regressor, save_dir: str, seed: int):
    """
    Saves the final trained classifier and regressor models to the correct subdirectories
    within the provided strategy directory.
    
    Args:
        classifier: The trained classifier model object.
        regressor: The trained regressor model object.
        save_dir (str): The full path to the specific strategy's deployment directory.
        seed (int): The seed used for this model training run.
    """
    # Create the subdirectories for the model weights
    clf_save_dir = os.path.join(save_dir, "classifier_weights")
    os.makedirs(clf_save_dir, exist_ok=True)
    
    reg_save_dir = os.path.join(save_dir, "regressor_weights")
    os.makedirs(reg_save_dir, exist_ok=True)

    # Define descriptive filenames for each model, including the seed
    clf_filename = f"final_clf_seed{seed}.joblib"
    reg_filename = f"final_reg_seed{seed}.joblib"
    clf_path = os.path.join(clf_save_dir, clf_filename)
    reg_path = os.path.join(reg_save_dir, reg_filename)

    # Save the models using joblib
    joblib.dump(classifier, clf_path)
    if regressor:
        joblib.dump(regressor, reg_path)

# In src/training/train_final_model.py

import os
import time
from datetime import timedelta
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import itertools
from tqdm import tqdm
import joblib

# Import helpers from the new dedicated helper file
from .utils.train_final_model_helpers import (
    load_data,
    load_selected_features,
    save_final_models
)

class FinalModelTrainer:
    def __init__(self):
        base = os.path.join(os.path.dirname(__file__), '../../data')
        self.features_file = os.path.join(base, 'final/features_final.xlsx')
        self.targets_file  = os.path.join(base, 'final/targets_final.xlsx')
        self.sel_file      = os.path.join(base, 'analysis/feature_selection/all_selected_features.xlsx')
        self.models_dir    = os.path.join(base, 'models/final_deployment') # Dedicated directory for final models

    def _get_full_training_data(self, category: str, timepoint: str, threshold_pct: int, top_n: int):
        """Prepares the entire dataset for final model training."""
        continuous_target_name = f"{category}_{timepoint}_raw"
        strategy_name = f"{category}_{timepoint}_binary_{threshold_pct}pct"
        
        features_df, targets_dict = load_data(self.features_file, self.targets_file)
        
        if continuous_target_name not in targets_dict:
            raise ValueError(f"Target '{continuous_target_name}' not found.")
            
        selected_features = load_selected_features(self.sel_file, category, strategy_name, top_n)
        if not selected_features:
            raise ValueError(f"No features found for strategy '{strategy_name}'.")

        X_base = features_df[['Ticker', 'Filing Date'] + selected_features]
        merged = pd.merge(X_base, targets_dict[continuous_target_name], on=['Ticker', 'Filing Date'], how='inner')
        merged.dropna(subset=[continuous_target_name], inplace=True)
        
        if merged.empty:
            raise ValueError("No data available after merging features and targets.")
            
        # Sort by date to ensure data is handled chronologically, though not splitting here
        merged = merged.sort_values(by='Filing Date').reset_index(drop=True)
        
        y_binary = (merged[continuous_target_name] >= (threshold_pct / 100.0)).astype(int)
        
        print(f"Prepared full dataset of {len(merged)} samples for final training.")
        return merged[selected_features], y_binary, merged[continuous_target_name]

    def _train_single_model_pair(self, X_train, y_bin_train, y_cont_train, seed):
        """Trains one pair of classifier and regressor for a given seed."""
        print(f"  - Training models for seed {seed}...")
        clf = RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=5, class_weight='balanced', random_state=seed, n_jobs=-1)
        clf.fit(X_train, y_bin_train)
        
        reg = None
        pos_train_idx = X_train.index[y_bin_train == 1]
        if not pos_train_idx.empty:
            reg = RandomForestRegressor(n_estimators=200, max_depth=8, min_samples_leaf=15, random_state=seed, n_jobs=-1)
            reg.fit(X_train.loc[pos_train_idx], y_cont_train.loc[pos_train_idx])
            
        return clf, reg

    def run(self, category: str, timepoint: str, threshold_pct: int, model_type: str, top_n: int, seeds: list):
        """
        Trains and saves the final deployment models for a single, winning strategy.
        This uses the entire historical dataset for training across multiple seeds.
        """
        start_time = time.time()
        print(f"\n### START ### Training Final Deployment Models ###")
        print(f"Strategy: {model_type} | Category: {category} | Timepoint: {timepoint} | Threshold: {threshold_pct}% | Top N: {top_n}")
        
        # 1. Load the full historical dataset
        X_full, y_binary_full, y_continuous_full = self._get_full_training_data(
            category, timepoint, threshold_pct, top_n
        )
        
        # 2. Train a model for each specified seed
        for seed in tqdm(seeds, desc="Training Final Models Across Seeds"):
            classifier, regressor = self._train_single_model_pair(
                X_full, y_binary_full, y_continuous_full, seed
            )
            
            # 3. Save each trained model pair
            save_final_models(
                classifier, regressor, self.models_dir, model_type, category, 
                timepoint, threshold_pct, top_n, seed
            )
            
        elapsed_time = timedelta(seconds=int(time.time() - start_time))
        print(f"\n### END ### All final models trained and saved in {elapsed_time}.")
        print(f"Models are located in: {os.path.join(self.models_dir, f'{model_type}_{category}_{timepoint}_{threshold_pct}pct_top{top_n}')}")

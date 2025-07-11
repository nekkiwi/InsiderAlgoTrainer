# In src/training/train_final_model.py

import os
import time
from datetime import timedelta
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

# Import helpers - note we no longer need load_selected_features
from .utils.train_final_model_helpers import load_data, save_final_models
# We reuse the robust feature selector from our backtesting code
from src.training.utils.feature_selector_helpers import select_features_for_fold

class FinalModelTrainer:
    def __init__(self):
        base = os.path.join(os.path.dirname(__file__), '../../data')
        self.features_file = os.path.join(base, 'final/features_final.xlsx')
        self.targets_file  = os.path.join(base, 'final/targets_final.xlsx')
        # We no longer need sel_file
        self.models_dir    = os.path.join(base, 'models/final_deployment')

    def _prepare_full_dataset(self, category: str, timepoint: str, threshold_pct: int):
        """Prepares the entire dataset for final model training from scratch."""
        features_df, targets_dict = load_data(self.features_file, self.targets_file)
        continuous_target_name = f"{category}_{timepoint}_raw"
        if continuous_target_name not in targets_dict:
            raise ValueError(f"Target '{continuous_target_name}' not found.")
            
        # Start with all features; they will be selected later
        all_features = features_df.drop(columns=['Ticker', 'Filing Date'], errors='ignore').columns.tolist()
        X_base = features_df[['Ticker', 'Filing Date'] + all_features]
        
        merged = pd.merge(X_base, targets_dict[continuous_target_name], on=['Ticker', 'Filing Date'], how='inner')
        merged.dropna(subset=[continuous_target_name], inplace=True)
        merged = merged.sort_values(by='Filing Date').reset_index(drop=True)
        
        y_binary = (merged[continuous_target_name] >= (threshold_pct / 100.0)).astype(int)
        
        return merged[all_features], y_binary, merged[continuous_target_name]

    def _train_single_model_pair(self, X_train, y_bin_train, y_cont_train, seed):
        # This function is correct and remains unchanged.
        # ... (your existing code) ...
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
        start_time = time.time()
        print(f"\n### START ### Training Final Deployment Models ###")
        print(f"Strategy: {model_type} | Category: {category} | Timepoint: {timepoint} | Threshold: {threshold_pct}% | Top N: {top_n}")

        # 1. Prepare the full, unscaled dataset with all features
        X_full, y_binary_full, y_continuous_full = self._prepare_full_dataset(
            category, timepoint, threshold_pct
        )
        
        # --- 2. PERFORM FINAL SCALING (CLIPPING & NORMALIZATION) ---
        continuous_features = [f for f in X_full.columns if X_full[f].nunique() > 2]
        
        print("- Clipping and scaling the full dataset...")
        for col in continuous_features:
            lower_bound = X_full[col].quantile(0.01)
            upper_bound = X_full[col].quantile(0.99)
            X_full[col].clip(lower_bound, upper_bound, inplace=True)
            
        final_scaler = MinMaxScaler()
        X_full[continuous_features] = final_scaler.fit_transform(X_full[continuous_features])
        
        # --- 3. PERFORM FINAL FEATURE SELECTION ---
        print(f"- Selecting top {top_n} features from the full dataset...")
        final_selected_features = select_features_for_fold(X_full, y_binary_full, top_n)
        X_final_sel = X_full[final_selected_features]

        # --- 4. SAVE THE INFERENCE ARTIFACTS (SCALER AND FEATURE LIST) ---
        # Create the specific directory for this strategy's artifacts
        strategy_dir_name = f"{model_type}_{category}_{timepoint}_{threshold_pct}pct_top{top_n}"
        save_dir = os.path.join(self.models_dir, strategy_dir_name)
        os.makedirs(save_dir, exist_ok=True)
        
        joblib.dump(final_scaler, os.path.join(save_dir, 'final_scaler.joblib'))
        joblib.dump(final_selected_features, os.path.join(save_dir, 'final_features.joblib'))
        print(f"- Saved final scaler and feature list to '{strategy_dir_name}'.")

        # 5. Train and save a model for each seed
        for seed in tqdm(seeds, desc="Training Final Models Across Seeds"):
            classifier, regressor = self._train_single_model_pair(
                X_final_sel, y_binary_full, y_continuous_full, seed
            )
            save_final_models(
                classifier, regressor, self.models_dir, model_type, category, 
                timepoint, threshold_pct, top_n, seed
            )

        elapsed_time = timedelta(seconds=int(time.time() - start_time))
        print(f"\n### END ### All final models trained and saved in {elapsed_time}.")
        print(f"All artifacts are located in: {save_dir}")

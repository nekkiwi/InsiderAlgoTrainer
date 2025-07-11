# In src/training/train.py

import os
import time
from datetime import timedelta
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import KFold
import itertools
from tqdm import tqdm

# --- Import the new feature selection helper ---
from src.training.utils.feature_selector_helpers import select_features_for_fold

from .utils.train_helpers import (
    load_data,
    # load_selected_features is NO LONGER needed here
    find_optimal_threshold,
    evaluate_test_fold,
    save_strategy_results,
    plot_final_summary
)


class ModelTrainer:
    def __init__(self):
        base = os.path.join(os.path.dirname(__file__), '../../data')
        self.features_file = os.path.join(base, 'final/features_final.xlsx')
        self.targets_file = os.path.join(base, 'final/targets_final.xlsx')
        self.stats_dir = os.path.join(base, 'training/stats')
        self.models_dir = os.path.join(base, 'models')

    def load_data(self):
        print("[LOAD] Loading and preparing data...")
        self.features_df, self.targets_dict = load_data(self.features_file, self.targets_file)
        print("[LOAD] Completed.")

    def _prepare_strategy_data(self, category: str, timepoint: str, threshold_pct: int):
        """
        Prepares the full X and y data for a single strategy without pre-selecting features.
        """
        continuous_target_name = f"{category}_{timepoint}_raw"
        if continuous_target_name not in self.targets_dict:
            return None, None, None
        
        # We start with ALL features
        all_features = self.features_df.drop(columns=['Ticker', 'Filing Date'], errors='ignore').columns.tolist()
        X_base = self.features_df[['Ticker', 'Filing Date'] + all_features]

        merged = pd.merge(X_base, self.targets_dict[continuous_target_name], on=['Ticker', 'Filing Date'], how='inner')
        merged.dropna(subset=[continuous_target_name], inplace=True)
        if merged.empty:
            return None, None, None
        
        merged = merged.sort_values(by='Filing Date').reset_index(drop=True)
        y_binary = (merged[continuous_target_name] >= (threshold_pct / 100.0)).astype(int)
        
        return merged[all_features], y_binary, merged[continuous_target_name]

    def _train_models(self, X_tr, y_bin_tr, y_cont_tr, seed):
        # This function remains the same
        clf = RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=5, class_weight='balanced', random_state=seed, n_jobs=-1)
        clf.fit(X_tr, y_bin_tr)
        reg = None
        pos_train_idx = X_tr.index[y_bin_tr == 1]
        if not pos_train_idx.empty:
            reg = RandomForestRegressor(n_estimators=200, max_depth=8, min_samples_leaf=15, random_state=seed, n_jobs=-1)
            reg.fit(X_tr.loc[pos_train_idx], y_cont_tr.loc[pos_train_idx])
        return clf, reg

    def run(self, category: str, timepoints: list, thresholds: list, model_type: str,
            optimize_for: str, top_n: int, seeds: list, n_splits: int = 5):
        start_time = time.time()
        print(f"\n### START ### {model_type} Walk-Forward with In-Loop Feature Selection ###")
        self.load_data()

        all_fold_results = []
        all_combinations = list(itertools.product(seeds, timepoints, thresholds))

        for seed, tp, thresh_pct in tqdm(all_combinations, desc="Processing Strategies"):
            # 1. Prepare data for the strategy with ALL features
            X, y_binary, y_continuous = self._prepare_strategy_data(category, tp, thresh_pct)
            if X is None: continue

            kf = KFold(n_splits=n_splits, shuffle=False)
            fold_count = 0

            for train_val_indices, test_indices in kf.split(X):
                fold_count += 1
                val_size = int(len(train_val_indices) * 0.2)
                train_indices, val_indices = train_val_indices[:-val_size], train_val_indices[-val_size:]

                # Split data for the current fold
                X_tr, y_bin_tr = X.iloc[train_indices], y_binary.iloc[train_indices]
                X_val, y_cont_val = X.iloc[val_indices], y_continuous.iloc[val_indices]
                X_ts, y_bin_ts, y_cont_ts = X.iloc[test_indices], y_binary.iloc[test_indices], y_continuous.iloc[test_indices]

                # 2. Perform feature selection USING ONLY THE TRAINING DATA FOR THIS FOLD
                selected_features = select_features_for_fold(X_tr, y_bin_tr, top_n)
                
                if not selected_features:
                    print(f"Fold {fold_count}: No features selected. Skipping.")
                    continue

                # 3. Use the selected features to train and evaluate
                X_tr_sel = X_tr[selected_features]
                X_val_sel = X_val[selected_features]
                X_ts_sel = X_ts[selected_features]
                y_cont_tr = y_continuous.iloc[train_indices]

                if X_tr_sel.empty or X_val_sel.empty or X_ts_sel.empty: continue

                classifier, regressor = self._train_models(X_tr_sel, y_bin_tr, y_cont_tr, seed)
                if regressor is None: continue

                val_buy_signals = classifier.predict(X_val_sel)
                val_pos_idx = X_val_sel.index[val_buy_signals == 1]
                if val_pos_idx.empty: continue
                
                val_predicted_returns = pd.Series(regressor.predict(X_val_sel.loc[val_pos_idx]), index=val_pos_idx)
                optimization_results = find_optimal_threshold(val_predicted_returns, y_cont_val.loc[val_pos_idx], optimize_for=optimize_for)
                optimal_X = optimization_results['optimal_threshold']

                fold_metrics = evaluate_test_fold(classifier, regressor, optimal_X, X_ts_sel, y_bin_ts, y_cont_ts)

                if fold_metrics:
                    result = {'Timepoint': tp, 'Threshold': f"{thresh_pct}%", 'Top_n_Features': top_n, 'Seed': seed, 'Fold': fold_count}
                    result.update(fold_metrics)
                    all_fold_results.append(result)
        
        # The rest of the script for saving and plotting results remains largely the same...
        if all_fold_results:
            results_df = pd.DataFrame(all_fold_results)
            save_strategy_results(results_df, self.stats_dir, model_type, category, optimize_for)
            summary_dir = os.path.join(self.stats_dir, 'summary')
            plot_final_summary(results_df, summary_dir, category, optimize_for)
            
            print(f"\n--- Strategy Performance (Mean over {n_splits} folds, Ranked by {optimize_for}) ---")
            
            # --- MODIFIED: display_cols now includes p-values ---
            display_cols = [
                'Adjusted Sharpe (Final)', 'Adjusted Sharpe (Classifier)', 
                'P-Value (Final vs Neg)', 'P-Value (Classifier vs Neg)',
                'Num Signals (Final)', 'Num Signals (Classifier)', 'Optimal Threshold'
            ]
            
            grouped = results_df.groupby(['Timepoint', 'Threshold', 'Top_n_Features'])[display_cols]
            mean_df = grouped.mean()
            std_df = grouped.std()

            formatted_df = pd.DataFrame(index=mean_df.index)
            for col in display_cols:
                if len(seeds) == 1:
                    formatted_df[col] = mean_df[col].map('{:.4f}'.format)
                else:
                    formatted_df[col] = (mean_df[col].map('{:.4f}'.format) + ' \u00B1 ' + std_df[col].fillna(0).map('{:.4f}'.format))

            sort_col = 'Adjusted Sharpe (Final)' if optimize_for == 'adjusted_sharpe' else 'Sharpe Ratio (Strategy)'
            sorted_index = mean_df[sort_col].sort_values(ascending=False).index
            print(formatted_df.loc[sorted_index].to_string())
        else:
            print("\n[INFO] No valid strategies were found after walk-forward validation.")
        
        elapsed_time = timedelta(seconds=int(time.time() - start_time))
        print(f"\n### END ### Total run time: {elapsed_time}")

        return pd.DataFrame(all_fold_results)


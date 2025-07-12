# In src/training/train.py

import os
import time
from datetime import timedelta
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import itertools
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit

# --- Import the new feature selection helper ---
from src.training.utils.feature_selector_helpers import select_features_for_fold

from .utils.train_helpers import (
    load_data,
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
        Prepares the full X and y data for a single strategy.
        """
        continuous_target_name = f"{category}_{timepoint}_raw"
        if continuous_target_name not in self.targets_dict:
            print(f"  [WARN] Target '{continuous_target_name}' not found. Skipping.")
            return None, None, None
        
        all_features = self.features_df.drop(columns=['Ticker', 'Filing Date'], errors='ignore').columns.tolist()
        X_base = self.features_df[['Ticker', 'Filing Date'] + all_features]

        merged = pd.merge(X_base, self.targets_dict[continuous_target_name], on=['Ticker', 'Filing Date'], how='inner')
        merged.dropna(subset=[continuous_target_name], inplace=True)
        if merged.empty:
            print(f"  [WARN] No data after merging for target '{continuous_target_name}'. Skipping.")
            return None, None, None
        
        merged = merged.sort_values(by='Filing Date').reset_index(drop=True)
        y_binary = (merged[continuous_target_name] >= (threshold_pct / 100.0)).astype(int)
        
        return merged[all_features], y_binary, merged[continuous_target_name]

    def _train_models(self, X_tr, y_bin_tr, y_cont_tr, seed):
        """
        Trains the classifier and regressor models.
        """
        # print("    [MODEL] Training classifier...")
        clf = RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=5, class_weight='balanced', random_state=seed, n_jobs=-1)
        clf.fit(X_tr, y_bin_tr)
        
        reg = None
        pos_train_idx = X_tr.index[y_bin_tr == 1]
        if not pos_train_idx.empty:
            # print(f"    [MODEL] Training regressor on {len(pos_train_idx)} positive samples...")
            reg = RandomForestRegressor(n_estimators=200, max_depth=8, min_samples_leaf=15, random_state=seed, n_jobs=-1)
            reg.fit(X_tr.loc[pos_train_idx], y_cont_tr.loc[pos_train_idx])
        else:
            pass
            # print("    [MODEL] No positive samples in training set for regressor.")
            
        return clf, reg

    def run(self, category: str, timepoints: list, thresholds: list, model_type: str,
            optimize_for: str, top_n: int, seeds: list, n_splits: int = 5):
        start_time = time.time()
        print(f"\n### START ### {model_type} Walk-Forward with In-Loop Feature Selection ###")
        self.load_data()
        
        all_feature_names = self.features_df.drop(columns=['Ticker', 'Filing Date'], errors='ignore').columns.tolist()
        continuous_features = [f for f in all_feature_names if self.features_df[f].nunique() > 2]
        print(f"[INFO] Identified {len(continuous_features)} continuous features for scaling.")

        all_fold_results = []
        all_combinations = list(itertools.product(seeds, timepoints, thresholds))
        print(f"[INFO] Starting walk-forward validation for {len(all_combinations)} strategy combinations.")

        for seed, tp, thresh_pct in tqdm(all_combinations, desc="Processing Strategies"):
            # print(f"\n[STRATEGY] Processing Seed: {seed}, Timepoint: {tp}, Threshold: {thresh_pct}%")
            X, y_binary, y_continuous = self._prepare_strategy_data(category, tp, thresh_pct)
            if X is None:
                continue

            tscv = TimeSeriesSplit(n_splits=n_splits)
            fold_count = 0

            for train_val_indices, test_indices in tscv.split(X):
                fold_count += 1
                # print(f"  [FOLD {fold_count}/{n_splits}]")
                val_size = int(len(train_val_indices) * 0.2)
                train_indices, val_indices = train_val_indices[:-val_size], train_val_indices[-val_size:]

                X_tr, y_bin_tr = X.iloc[train_indices].copy(), y_binary.iloc[train_indices].copy()
                X_val, y_cont_val = X.iloc[val_indices].copy(), y_continuous.iloc[val_indices].copy()
                X_ts, y_bin_ts, y_cont_ts = X.iloc[test_indices].copy(), y_binary.iloc[test_indices].copy(), y_continuous.iloc[test_indices].copy()
                y_cont_tr = y_continuous.iloc[train_indices].copy()
                # print(f"    [DATA] Split sizes: Train={len(X_tr)}, Val={len(X_val)}, Test={len(X_ts)}")

                # --- Point-in-time Scaling and Clipping ---
                # print("    [PREP] Clipping and scaling data...")
                for col in continuous_features:
                    lower_bound = X_tr[col].quantile(0.01)
                    upper_bound = X_tr[col].quantile(0.99)
                    X_tr[col] = X_tr[col].clip(lower_bound, upper_bound)
                    X_val[col] = X_val[col].clip(lower_bound, upper_bound)
                    X_ts[col] = X_ts[col].clip(lower_bound, upper_bound)

                scaler = MinMaxScaler()
                scaler.fit(X_tr[continuous_features])
                X_tr[continuous_features] = scaler.transform(X_tr[continuous_features])
                X_val[continuous_features] = scaler.transform(X_val[continuous_features])
                X_ts[continuous_features] = scaler.transform(X_ts[continuous_features])
                
                # --- Feature Selection ---
                # print(f"    [FEATSEL] Selecting top {top_n} features...")
                selected_features = select_features_for_fold(X_tr, y_bin_tr, top_n)
                if not selected_features:
                    # print("    [FEATSEL] No features selected. Skipping fold.")
                    continue
                # print(f"    [FEATSEL] Selected {len(selected_features)} features.")

                X_tr_sel, X_val_sel, X_ts_sel = X_tr[selected_features], X_val[selected_features], X_ts[selected_features]

                classifier, regressor = self._train_models(X_tr_sel, y_bin_tr, y_cont_tr, seed)
                if regressor is None:
                    # print("    [MODEL] Regressor not trained. Skipping fold.")
                    continue

                # --- Validation and Optimization ---
                val_buy_signals = classifier.predict(X_val_sel)
                val_pos_idx = X_val_sel.index[val_buy_signals == 1]
                if val_pos_idx.empty:
                    # print("    [VALIDATE] No buy signals on validation set. Skipping fold.")
                    continue
                # print(f"    [VALIDATE] Found {len(val_pos_idx)} buy signals on validation set.")
                
                val_predicted_returns = pd.Series(regressor.predict(X_val_sel.loc[val_pos_idx]), index=val_pos_idx)
                optimization_results = find_optimal_threshold(val_predicted_returns, y_cont_val.loc[val_pos_idx], optimize_for=optimize_for)
                optimal_X = optimization_results['optimal_threshold']
                # print(f"    [VALIDATE] Optimal regressor threshold found: {optimal_X:.4f}")

                # --- Test Set Evaluation ---
                fold_metrics = evaluate_test_fold(classifier, regressor, optimal_X, X_ts_sel, y_bin_ts, y_cont_ts)
                if fold_metrics:
                    # print(f"    [EVAL] Test Adj. Sharpe: {fold_metrics.get('Adjusted Sharpe (Final)', 'N/A'):.4f}, Num Signals: {fold_metrics.get('Num Signals (Final)', 'N/A')}")
                    result = {'Timepoint': tp, 'Threshold': f"{thresh_pct}%", 'Top_n_Features': top_n, 'Seed': seed, 'Fold': fold_count}
                    result.update(fold_metrics)
                    all_fold_results.append(result)
        
        if all_fold_results:
            results_df = pd.DataFrame(all_fold_results)
            save_strategy_results(results_df, self.stats_dir, model_type, category, optimize_for)
            plot_final_summary(results_df, self.stats_dir, category, optimize_for)
            
            print(f"\n--- Strategy Performance (Mean over {n_splits} folds, Ranked by {optimize_for}) ---")
            
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
                    formatted_df[col] = (mean_df[col].map('{:.4f}'.format) + ' ± ' + std_df[col].fillna(0).map('{:.4f}'.format))

            sort_col = 'Adjusted Sharpe (Final)' if optimize_for == 'adjusted_sharpe' else 'Sharpe Ratio (Strategy)'
            sorted_index = mean_df[sort_col].sort_values(ascending=False).index
            print(formatted_df.loc[sorted_index].to_string())
        else:
            print("\n[INFO] No valid strategies were found after walk-forward validation.")
        
        elapsed_time = timedelta(seconds=int(time.time() - start_time))
        print(f"\n### END ### Total run time: {elapsed_time}")

        return pd.DataFrame(all_fold_results)

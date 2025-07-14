# In src/training/train.py

import os
import time
from datetime import timedelta
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
# Note: You will need to install lightgbm: pip install lightgbm
from lightgbm import LGBMClassifier, LGBMRegressor
import itertools
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

from src.training.utils.feature_selector_helpers import select_features_for_fold
from .utils.train_helpers import (
    load_data,
    find_optimal_threshold,
    evaluate_test_fold,
    save_strategy_results
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
        continuous_target_name = f"{category}_{timepoint}_raw"
        if continuous_target_name not in self.targets_dict:
            return None, None, None
        
        all_features = self.features_df.drop(columns=['Ticker', 'Filing Date'], errors='ignore').columns.tolist()
        X_base = self.features_df[['Ticker', 'Filing Date'] + all_features]

        merged = pd.merge(X_base, self.targets_dict[continuous_target_name], on=['Ticker', 'Filing Date'], how='inner')
        merged.dropna(subset=[continuous_target_name], inplace=True)
        if merged.empty: return None, None, None
        
        merged = merged.sort_values(by='Filing Date').reset_index(drop=True)
        y_binary = (merged[continuous_target_name] >= (threshold_pct / 100.0)).astype(int)
        
        return merged[all_features], y_binary, merged[continuous_target_name]

    def _get_model_and_params(self, model_type: str, model_class: str):
        """Returns the model instance and parameter grid for a given model type and class (Classifier/Regressor)."""
        if model_type == 'RandomForest':
            if model_class == 'Classifier':
                model = RandomForestClassifier(random_state=42, n_jobs=-1)
                param_dist = {
                    'n_estimators': [100, 200, 300], 'max_depth': [5, 10, 15, None],
                    'min_samples_leaf': [5, 10, 20], 'max_features': ['sqrt', 0.5, 0.7],
                    'class_weight': ['balanced', None]
                }
            else: # Regressor
                model = RandomForestRegressor(random_state=42, n_jobs=-1)
                param_dist = {
                    'n_estimators': [100, 200, 300], 'max_depth': [5, 8, 12, None],
                    'min_samples_leaf': [10, 20, 30], 'max_features': ['sqrt', 0.5, 0.7]
                }
        elif model_type == 'LightGBM':
            if model_class == 'Classifier':
                model = LGBMClassifier(random_state=42, n_jobs=-1, verbosity=-1)
                param_dist = {
                    'n_estimators': [100, 200, 300], 'learning_rate': [0.01, 0.05, 0.1],
                    'num_leaves': [31, 50, 70], 'max_depth': [5, 10, -1],
                    'reg_alpha': [0.0, 0.1, 0.5], 'reg_lambda': [0.0, 0.1, 0.5]
                }
            else: # Regressor
                model = LGBMRegressor(random_state=42, n_jobs=-1, verbosity=-1)
                param_dist = {
                    'n_estimators': [100, 200, 300], 'learning_rate': [0.01, 0.05, 0.1],
                    'num_leaves': [31, 50, 70], 'max_depth': [5, 10, -1],
                    'reg_alpha': [0.0, 0.1, 0.5], 'reg_lambda': [0.0, 0.1, 0.5]
                }
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        return model, param_dist

    def _train_models(self, X_tr, y_bin_tr, y_cont_tr, seed, model_type, use_hyperparameter_tuning, tscv_inner, fold_info):
        """
        Trains models, performing sequential hyperparameter tuning for the classifier and then the regressor.
        """
        best_classifier_params = 'default'
        best_regressor_params = 'default'

        # --- Stage 1: Train or Tune the Classifier ---
        if use_hyperparameter_tuning:
            print(f"    [TUNING CLF] Starting for {fold_info}...")
            clf_model, clf_param_dist = self._get_model_and_params(model_type, 'Classifier')
            random_search_clf = RandomizedSearchCV(
                estimator=clf_model, param_distributions=clf_param_dist, n_iter=15,
                cv=tscv_inner, scoring='roc_auc', random_state=seed, n_jobs=-1, verbose=1
            )
            random_search_clf.fit(X_tr, y_bin_tr)
            classifier = random_search_clf.best_estimator_
            best_classifier_params = random_search_clf.best_params_
        else:
            # Default model training
            if model_type == 'RandomForest':
                classifier = RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=5, class_weight='balanced', random_state=seed, n_jobs=-1)
            elif model_type == 'LightGBM':
                classifier = LGBMClassifier(n_estimators=100, learning_rate=0.1, num_leaves=31, random_state=seed, n_jobs=-1, verbosity=-1)
            classifier.fit(X_tr, y_bin_tr)

        # --- Stage 2: Train or Tune the Regressor ---
        regressor = None
        pos_train_idx = X_tr.index[y_bin_tr == 1]
        if not pos_train_idx.empty:
            X_reg_train = X_tr.loc[pos_train_idx]
            y_reg_train = y_cont_tr.loc[pos_train_idx]

            if use_hyperparameter_tuning:
                print(f"    [TUNING REG] Starting for {fold_info}...")
                reg_model, reg_param_dist = self._get_model_and_params(model_type, 'Regressor')
                random_search_reg = RandomizedSearchCV(
                    estimator=reg_model, param_distributions=reg_param_dist, n_iter=15,
                    cv=tscv_inner, scoring='neg_mean_squared_error', random_state=seed, n_jobs=-1, verbose=1
                )
                random_search_reg.fit(X_reg_train, y_reg_train)
                regressor = random_search_reg.best_estimator_
                best_regressor_params = random_search_reg.best_params_
            else:
                # Default regressor training
                if model_type == 'RandomForest':
                    regressor = RandomForestRegressor(n_estimators=200, max_depth=8, min_samples_leaf=15, random_state=seed, n_jobs=-1)
                elif model_type == 'LightGBM':
                    regressor = LGBMRegressor(n_estimators=100, learning_rate=0.1, num_leaves=31, random_state=seed, n_jobs=-1, verbosity=-1)
                regressor.fit(X_reg_train, y_reg_train)

        return classifier, regressor, best_classifier_params, best_regressor_params

    def run(self, category: str, timepoints: list, thresholds: list, model_type: str, top_n: int, seeds: list, use_hyperparameter_tuning: bool = False):
        start_time = time.time()
        print(f"\n### START ### {model_type} Walk-Forward (Tuning: {use_hyperparameter_tuning}) ###")
        self.load_data()
        
        all_feature_names = self.features_df.drop(columns=['Ticker', 'Filing Date'], errors='ignore').columns.tolist()
        continuous_features = [f for f in all_feature_names if self.features_df[f].nunique() > 2]
        
        all_fold_results = []
        all_combinations = list(itertools.product(seeds, timepoints, thresholds))

        print(f"[INFO] Starting validation for {len(all_combinations)} strategy combinations.")

        for seed, tp, thresh_pct in tqdm(all_combinations, desc="Processing Strategies"):
                
            X, y_binary, y_continuous = self._prepare_strategy_data(category, tp, thresh_pct)
            if X is None: continue

            tscv_outer = TimeSeriesSplit(n_splits=5)
            fold_count = 0

            for train_val_indices, test_indices in tscv_outer.split(X):
                fold_count += 1
                val_size = int(len(train_val_indices) * 0.2)
                train_indices, val_indices = train_val_indices[:-val_size], train_val_indices[-val_size:]

                X_tr, y_bin_tr = X.iloc[train_indices].copy(), y_binary.iloc[train_indices].copy()
                X_val, y_cont_val = X.iloc[val_indices].copy(), y_continuous.iloc[val_indices].copy()
                X_ts, y_bin_ts, y_cont_ts = X.iloc[test_indices].copy(), y_binary.iloc[test_indices].copy(), y_continuous.iloc[test_indices].copy()
                y_cont_tr = y_continuous.iloc[train_indices].copy()

                # Pre-processing...
                pd.set_option('future.no_silent_downcasting', True)
                for col in continuous_features:
                    lower = X_tr[col].quantile(0.01); upper = X_tr[col].quantile(0.99)
                    X_tr[col] = X_tr[col].clip(lower, upper); X_val[col] = X_val[col].clip(lower, upper); X_ts[col] = X_ts[col].clip(lower, upper)
                
                scaler = MinMaxScaler().fit(X_tr[continuous_features])
                X_tr[continuous_features] = scaler.transform(X_tr[continuous_features])
                X_val[continuous_features] = scaler.transform(X_val[continuous_features])
                X_ts[continuous_features] = scaler.transform(X_ts[continuous_features])
                
                selected_features = select_features_for_fold(X_tr, y_bin_tr, top_n, seed)
                if not selected_features: continue
                X_tr_sel, X_val_sel, X_ts_sel = X_tr[selected_features], X_val[selected_features], X_ts[selected_features]
                
                tscv_inner = TimeSeriesSplit(n_splits=3)
                fold_info_str = f"Strategy {tp}-{thresh_pct}%, Fold {fold_count}, Seed {seed}"
                
                # --- Updated to capture both sets of best params ---
                classifier, regressor, best_clf_params, best_reg_params = self._train_models(
                    X_tr_sel, y_bin_tr, y_cont_tr, seed, model_type, use_hyperparameter_tuning, tscv_inner, fold_info_str
                )
                if regressor is None: continue
                
                # Validation and evaluation...
                val_buy_signals = classifier.predict(X_val_sel)
                val_pos_idx = X_val_sel.index[val_buy_signals == 1]
                if val_pos_idx.empty: continue
                
                val_predicted_returns = pd.Series(regressor.predict(X_val_sel.loc[val_pos_idx]), index=val_pos_idx)
                optimization_results = find_optimal_threshold(val_predicted_returns, y_cont_val.loc[val_pos_idx])
                optimal_X = optimization_results['optimal_threshold']

                fold_metrics = evaluate_test_fold(classifier, regressor, optimal_X, X_ts_sel, y_bin_ts, y_cont_ts)
                
                if fold_metrics:
                    result = {
                        'Timepoint': tp, 'Threshold': f"{thresh_pct}%", 
                        'Seed': seed, 'Fold': fold_count, 'Model': model_type, 
                        'Best Classifier Params': str(best_clf_params),
                        'Best Regressor Params': str(best_reg_params)
                    }
                    result.update(fold_metrics)
                    all_fold_results.append(result)
    
        if all_fold_results:
            results_df = pd.DataFrame(all_fold_results)
            save_strategy_results(results_df, self.stats_dir, f"{model_type}_{'Tuned' if use_hyperparameter_tuning else 'Default'}", category)
        
        elapsed_time = timedelta(seconds=int(time.time() - start_time))
        print(f"\n### END ### Total run time: {elapsed_time}")
        return pd.DataFrame(all_fold_results)


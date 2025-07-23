# In src/training/train.py

import os
import time
from datetime import timedelta
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from lightgbm import LGBMClassifier, LGBMRegressor
import itertools
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

from .utils.train_helpers import (
    load_data,
    find_optimal_threshold,
    evaluate_test_fold,
    save_strategy_results,
    select_features_for_fold,
    convert_timepoints_to_bdays
)

class ModelTrainer:
    def __init__(self):
        base = os.path.join(os.path.dirname(__file__), '../../data')
        self.features_file = os.path.join(base, 'final/features_final.xlsx')
        self.stock_data_file = os.path.join(base, 'final/stock_data_final.xlsx')
        self.targets_file = os.path.join(base, 'final/targets_final.xlsx')
        self.stats_dir = os.path.join(base, 'training/stats')
        self.models_dir = os.path.join(base, 'models')

    def load_data(self):
        print("[LOAD] Loading and preparing data...")
        self.features_df, self.targets_dict = load_data(self.features_file, self.targets_file)
        print("[LOAD] Completed.")
        
    def _load_tpsl_data(self, category: str):
        """Loads the High and Low sheets for TP/SL calculation once."""
        print(f"[LOAD] Loading High/Low price data for '{category}' TP/SL calculations...")
        try:
            high_sheet_name = f"{category.capitalize()}_High"
            low_sheet_name = f"{category.capitalize()}_Low"
            
            self.high_sheet = pd.read_excel(self.stock_data_file, sheet_name=high_sheet_name)
            self.low_sheet = pd.read_excel(self.stock_data_file, sheet_name=low_sheet_name)
            
            # Pre-process dates to ensure correct matching later
            self.high_sheet['Filing Date'] = pd.to_datetime(self.high_sheet['Filing Date'], dayfirst=True)
            self.low_sheet['Filing Date'] = pd.to_datetime(self.low_sheet['Filing Date'], dayfirst=True)
            
            print("[LOAD] High/Low data loaded successfully.")
        except Exception as e:
            print(f"[WARN] Could not load High/Low sheets for TP/SL: {e}")
            self.high_sheet = None
            self.low_sheet = None

    def _prepare_strategy_data(self, category: str, timepoint: str, threshold_pct: int):
        """
        category: 'return' or 'alpha' (Close is implied)
        timepoint: e.g. '5d', '20d'
        """
        # Always use Return_Close or Alpha_Close sheets
        sheet_name = "Return_Close" if category.lower() == "return" else "Alpha_Close"
        column_name = f'{category.lower()}_{timepoint}'
        if sheet_name not in self.targets_dict:
            print(f"[ERROR] Target sheet '{sheet_name}' not found in targets_final.xlsx")
            return None, None, None
        target_df = self.targets_dict[sheet_name].copy()
        if column_name not in target_df.columns:
            print(f"[ERROR] Target column '{column_name}' not found in sheet '{sheet_name}'")
            return None, None, None

        all_features = self.features_df.drop(columns=['Ticker', 'Filing Date'], errors='ignore').columns.tolist()
        X_base = self.features_df[['Ticker', 'Filing Date'] + all_features]
        merged = pd.merge(X_base, target_df[['Ticker', 'Filing Date', column_name]],
                        on=['Ticker', 'Filing Date'], how='inner')
        merged.dropna(subset=[column_name], inplace=True)
        if merged.empty:
            print(f"[WARN] After merge/dropna, no data for {category} at timepoint {timepoint}")
            return None, None, None

        merged = merged.sort_values(by='Filing Date').reset_index(drop=True)
        y_continuous = merged[column_name]
        y_binary = (y_continuous >= (threshold_pct / 100.0)).astype(int)
        return merged[all_features], merged[['Ticker', 'Filing Date']], y_binary, y_continuous

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
        
        lgbm_stochastic_params = {
                'subsample': 0.8,
                'colsample_bytree': 0.8
            }

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
                classifier = LGBMClassifier(n_estimators=100,
                                            learning_rate=0.1, 
                                            num_leaves=31,
                                            random_state=seed, n_jobs=-1, verbosity=-1, **lgbm_stochastic_params)
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
                    regressor = LGBMRegressor(n_estimators=100,
                                              learning_rate=0.1,
                                              num_leaves=31,
                                              random_state=seed, n_jobs=-1, verbosity=-1, **lgbm_stochastic_params)
                regressor.fit(X_reg_train, y_reg_train)

        return classifier, regressor, best_classifier_params, best_regressor_params

    def run(self, category: str, timepoints: list, thresholds: list, model_type: str, top_n: int,
            seeds: list, tp_sl_configs, use_hyperparameter_tuning: bool = False):
        start_time = time.time()
        
        MIN_TEST_SET_SIZE = 30  # e.g., require at least 30 samples in the test set
        MIN_SIGNALS = 10
        
        print(f"\n### START ### {model_type} Walk-Forward (Tuning: {use_hyperparameter_tuning}) ###")
        self.load_data()
        self._load_tpsl_data(category)
        
        try:
            timepoint_map = convert_timepoints_to_bdays(timepoints)
            print(f"[INFO] Timepoint to business day map: {timepoint_map}")
        except ValueError as e:
            print(f"[ERROR] Could not parse timepoints: {e}")
            return # Exit if formats are invalid
        
        all_feature_names = self.features_df.drop(columns=['Ticker', 'Filing Date'], errors='ignore').columns.tolist()
        continuous_features = [f for f in all_feature_names if self.features_df[f].nunique() > 2]
        
        all_fold_results = []
        all_combinations = list(itertools.product(seeds, timepoints, thresholds, tp_sl_configs))

        print(f"[INFO] Starting validation for {len(all_combinations)} strategy combinations.")

        skipped_folds = 0

        for seed, tp, thresh_pct, tp_sl in tqdm(all_combinations, desc="Processing Strategies"):
            X, X_meta, y_binary, y_continuous = self._prepare_strategy_data(category, tp, thresh_pct)
            if X is None: 
                continue

            tscv_outer = TimeSeriesSplit(n_splits=5)
            fold_count = 0

            for train_val_indices, test_indices in tscv_outer.split(X):
                fold_count += 1
                fold_info_str = f"Strategy {tp}-{thresh_pct}%, Fold {fold_count}, Seed {seed}"
                print(f"\nNow processing: {fold_info_str}")
                
                # --- REFACTORED LOGIC ---

                print("\n1. Initial Data Splitting")
                # 1. Split the train-validation block into a dedicated training set and a validation set FIRST.
                # This prevents data leakage during feature selection and scaling.
                val_size = int(len(train_val_indices) * 0.2)
                if val_size < 1 and len(train_val_indices) > 1: val_size = 1

                if len(train_val_indices) <= val_size:
                    print(f"  [SKIP] Not enough data in train-val set ({len(train_val_indices)}) to create a validation set of size {val_size}.")
                    skipped_folds += 1
                    continue

                train_indices, val_indices = train_val_indices[:-val_size], train_val_indices[-val_size:]
                
                X_tr = X.iloc[train_indices].copy()
                y_bin_tr = y_binary.iloc[train_indices].copy()
                y_cont_tr = y_continuous.iloc[train_indices].copy()

                X_val = X.iloc[val_indices].copy()
                y_cont_val = y_continuous.iloc[val_indices].copy()

                X_ts = X.iloc[test_indices].copy()
                y_bin_ts = y_binary.iloc[test_indices].copy()
                y_cont_ts = y_continuous.iloc[test_indices].copy()
                X_meta_ts = X_meta.loc[test_indices].copy() # Ensure metadata is also split correctly
                print(f"  [SPLIT] Raw sizes -> Train: {len(X_tr)}, Val: {len(X_val)}, Test: {len(X_ts)}")

                if len(X_ts) < MIN_TEST_SET_SIZE:
                    print(f"  [SKIP] Test set size ({len(X_ts)}) is smaller than the minimum required ({MIN_TEST_SET_SIZE}). Skipping fold.")
                    skipped_folds += 1
                    continue

                print("\n2. Feature Selection (on Training set only)")
                # 2. Perform feature selection ONLY on the training data (X_tr) to prevent leakage.
                selected_features = select_features_for_fold(X_tr, y_bin_tr, top_n, seed)
                if not selected_features:
                    print(f"  [SKIP] No features selected for {fold_info_str}.")
                    skipped_folds += 1
                    continue
                print(f"  [SELECT] Selected {len(selected_features)} features: {', '.join(selected_features)}")

                # Filter all dataframes to only the selected features before further processing
                X_tr = X_tr[selected_features]
                X_val = X_val[selected_features]
                X_ts = X_ts[selected_features]

                print("\n3. Data Cleaning (based on selected features)")
                # 3. Drop rows with NaNs in the *selected features* from all sets.
                tr_pre_drop = len(X_tr)
                train_complete_indices = X_tr.dropna().index
                X_tr = X_tr.loc[train_complete_indices]
                y_bin_tr, y_cont_tr = y_bin_tr.loc[train_complete_indices], y_cont_tr.loc[train_complete_indices]
                print(f"  [CLEAN] Train Set: {tr_pre_drop} -> {len(X_tr)} ({tr_pre_drop - len(X_tr)} dropped)")
                
                val_pre_drop = len(X_val)
                val_complete_indices = X_val.dropna().index
                X_val = X_val.loc[val_complete_indices]
                y_cont_val = y_cont_val.loc[val_complete_indices]
                print(f"  [CLEAN] Validation Set: {val_pre_drop} -> {len(X_val)} ({val_pre_drop - len(X_val)} dropped)")

                ts_pre_drop = len(X_ts)
                test_complete_indices = X_ts.dropna().index
                X_ts = X_ts.loc[test_complete_indices]
                y_bin_ts, y_cont_ts = y_bin_ts.loc[test_complete_indices], y_cont_ts.loc[test_complete_indices]
                X_meta_ts = X_meta_ts.loc[test_complete_indices]
                print(f"  [CLEAN] Test Set: {ts_pre_drop} -> {len(X_ts)} ({ts_pre_drop - len(X_ts)} dropped)")
                
                if X_tr.empty or X_ts.empty or X_val.empty:
                    print(f"  [SKIP] Not enough data after NaN drop for {fold_info_str}.")
                    skipped_folds += 1
                    continue

                print("\n4. Data Preprocessing (Scaling & Clipping)")
                # 4. Fit scaler ONLY on the cleaned training data and transform all sets.
                pd.set_option('future.no_silent_downcasting', True)
                final_continuous_features = [f for f in continuous_features if f in X_tr.columns]
                
                for col in final_continuous_features:
                    lower, upper = X_tr[col].quantile(0.01), X_tr[col].quantile(0.99)
                    X_tr[col] = X_tr[col].clip(lower, upper)
                    X_val[col] = X_val[col].clip(lower, upper)
                    X_ts[col] = X_ts[col].clip(lower, upper)
                
                if final_continuous_features:
                    scaler = MinMaxScaler().fit(X_tr[final_continuous_features])
                    X_tr[final_continuous_features] = scaler.transform(X_tr[final_continuous_features])
                    X_val[final_continuous_features] = scaler.transform(X_val[final_continuous_features])
                    X_ts[final_continuous_features] = scaler.transform(X_ts[final_continuous_features])
                print("  [PREP] Scaling and clipping complete.")
                
                # --- END REFACTORED LOGIC ---
                print("\n5. Model Training")
                X_tr_sel, X_val_sel, X_ts_sel = X_tr[selected_features], X_val[selected_features], X_ts[selected_features]
                
                tscv_inner = TimeSeriesSplit(n_splits=3)
                classifier, regressor, best_clf_params, best_reg_params = self._train_models(
                    X_tr_sel, y_bin_tr, y_cont_tr, seed, model_type, use_hyperparameter_tuning, tscv_inner, fold_info_str
                )
                if regressor is None: 
                    continue
                
                print("\n6. Threshold Optimization")
                val_buy_signals = classifier.predict(X_val_sel)
                val_pos_idx = X_val_sel.index[val_buy_signals == 1]
                if val_pos_idx.empty: 
                    continue
                val_predicted_returns = pd.Series(regressor.predict(X_val_sel.loc[val_pos_idx]), index=val_pos_idx)
                optimization_results = find_optimal_threshold(val_predicted_returns, y_cont_val.loc[val_pos_idx])
                optimal_X = optimization_results['optimal_threshold']
                print(f"  [OPTIMIZE] Optimal threshold found on validation set: {optimal_X:.4f}")
                # TODO: CAN BE NAN FOR SOME REASON???

                print("\n7. Final Evaluation on Test Set")
                # Pass the stock_data_final_file to enable proper TP/SL evaluation
                fold_metrics = evaluate_test_fold(
                    classifier, regressor, optimal_X, X_ts_sel, y_bin_ts, y_cont_ts, category,
                    high_sheet=self.high_sheet,  # Pass the loaded DataFrame
                    low_sheet=self.low_sheet,    # Pass the loaded DataFrame
                    window_days=timepoint_map[tp],
                    tp_sl=tp_sl,
                    X_meta_ts=X_meta_ts
                )

                if not fold_metrics:
                    print(f"[SKIP] No valid fold metrics for {fold_info_str}")
                    skipped_folds += 1
                    continue
                
                num_signals_final = fold_metrics.get('Num Signals (Final)', 'N/A')
                if  num_signals_final < MIN_SIGNALS:
                    print(f"  [SKIP] Number of signals ({num_signals_final}) is smaller than the minimum required ({MIN_SIGNALS}). Skipping fold.")
                    skipped_folds += 1
                    continue
                
                # If the fold was not skipped, append results
                print(f"  [SUCCESS] Fold {fold_count} completed. Signals: {num_signals_final}")
                
                if fold_metrics:
                    adj_sharpe_final = fold_metrics.get('Adjusted Sharpe (Final)', 'N/A')
                    adj_sharpe_tpsl = fold_metrics.get('Adjusted Sharpe (TP/SL)', 'N/A')
                    
                    adj_sharpe_final_str = f"{adj_sharpe_final:.3f}" if isinstance(adj_sharpe_final, (int, float)) else str(adj_sharpe_final)
                    adj_sharpe_tpsl_str = f"{adj_sharpe_tpsl:.3f}" if isinstance(adj_sharpe_tpsl, (int, float)) else str(adj_sharpe_tpsl)
                    print(f"\n  *** FOLD RESULT -> Adj Sharpe (Final): {adj_sharpe_final_str}, Adj Sharpe (TP/SL): {adj_sharpe_tpsl_str} ***")
                    result = {
                        'Timepoint': tp, 'Threshold': f"{thresh_pct}%", 
                        'Seed': seed, 'Fold': fold_count, 'Model': model_type, 
                        'Best Classifier Params': str(best_clf_params),
                        'Best Regressor Params': str(best_reg_params),
                        'TP': tp_sl[0],
                        'SL': tp_sl[1]
                    }
                    result.update(fold_metrics)
                    all_fold_results.append(result)

        print(f"[INFO] {skipped_folds} folds were skipped due to no valid results.")
        print(f"[INFO] {len(all_fold_results)} folds produced results.")

        if all_fold_results:
            results_df = pd.DataFrame(all_fold_results)
            save_strategy_results(results_df, self.stats_dir, f"{model_type}_{'Tuned' if use_hyperparameter_tuning else 'Default'}", category)
        
        elapsed_time = timedelta(seconds=int(time.time() - start_time))
        print(f"\n### END ### Total run time: {elapsed_time}")
        return pd.DataFrame(all_fold_results)

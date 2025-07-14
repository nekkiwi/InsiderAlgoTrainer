import os
import time
from datetime import timedelta
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from .utils.train_helpers import select_features_for_fold
from .utils.train_final_model_helpers import load_data, save_final_models

class FinalModelTrainer:
    def __init__(self):
        """
        Initializes the trainer.
        Args:
            optimize_for (str): The metric used during walk-forward validation 
                                (e.g., 'adjusted_sharpe'), needed to locate the correct results file.
        """
        base = os.path.join(os.path.dirname(__file__), '../../data')
        self.features_file = os.path.join(base, 'final/features_final.xlsx')
        self.targets_file = os.path.join(base, 'final/targets_final.xlsx')
        self.models_dir = os.path.join(base, 'models/final_deployment')
        self.stats_dir = os.path.join(base, 'training/stats')

    def _get_optimal_threshold(self, model_type: str, category: str, timepoint: str, threshold_pct: int, top_n: int) -> float:
        """
        Loads the pre-calculated optimal threshold by averaging the results 
        from the walk-forward validation process.
        """
        results_file = os.path.join(self.stats_dir, f"{model_type}_Default_{category}_walk_forward_results.xlsx")
        if not os.path.exists(results_file):
            raise FileNotFoundError(f"Walk-forward results file not found: {results_file}")
            
        results_df = pd.read_excel(results_file, sheet_name="Per-Fold Summary")
        
        strategy_results = results_df[
            (results_df['Timepoint'] == timepoint) &
            (results_df['Threshold'] == f"{threshold_pct}%")
        ]
        
        if strategy_results.empty:
            raise ValueError(f"Strategy {model_type}-{category}-{timepoint}-{threshold_pct}% not found in results file.")
            
        optimal_threshold = strategy_results['Optimal Threshold (Mean)'].mean()
        print(f"- Found and loaded robust Optimal Threshold (X): {optimal_threshold:.4f}")
        return optimal_threshold

    def _prepare_full_dataset(self, category: str, timepoint: str, threshold_pct: int):
        """Prepares the entire dataset for final model training from the raw feature and target files."""
        features_df, targets_dict = load_data(self.features_file, self.targets_file)
        continuous_target_name = f"{category}_{timepoint}_raw"
        if continuous_target_name not in targets_dict:
            raise ValueError(f"Target '{continuous_target_name}' not found.")

        all_features = features_df.drop(columns=['Ticker', 'Filing Date'], errors='ignore').columns.tolist()
        merged = pd.merge(features_df, targets_dict[continuous_target_name], on=['Ticker', 'Filing Date'], how='inner')
        merged.dropna(subset=[continuous_target_name], inplace=True)
        merged = merged.sort_values(by='Filing Date').reset_index(drop=True)

        y_binary = (merged[continuous_target_name] >= (threshold_pct / 100.0)).astype(int)
        return merged[all_features], y_binary, merged[continuous_target_name]

    def _train_single_model_pair(self, X_train, y_bin_train, y_cont_train, seed, model_type: str):
        """Trains one pair of models (classifier and regressor) for a given seed."""
        if model_type == 'RandomForest':
            clf = RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=5, class_weight='balanced', random_state=seed, n_jobs=-1)
            reg_model = RandomForestRegressor(n_estimators=200, max_depth=8, min_samples_leaf=15, random_state=seed, n_jobs=-1)
        elif model_type == 'LightGBM':
            clf = LGBMClassifier(random_state=seed, n_jobs=-1, verbosity=-1)
            reg_model = LGBMRegressor(random_state=seed, n_jobs=-1, verbosity=-1)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        clf.fit(X_train, y_bin_train)
        
        reg = None
        pos_train_idx = X_train.index[y_bin_train == 1]
        if not pos_train_idx.empty:
            reg = reg_model
            reg.fit(X_train.loc[pos_train_idx], y_cont_train.loc[pos_train_idx])

        return clf, reg

    def run(self, category: str, timepoint: str, threshold_pct: int, model_type: str, top_n: int, seeds: list):
        """
        Executes the full pipeline to train and save final models and all related artifacts.
        """
        start_time = time.time()
        print(f"\n### START ### Training Final Deployment Models on ALL Data ###")
        print(f"Strategy: {model_type} | Category: {category} | Timepoint: {timepoint} | Threshold: {threshold_pct}% | Top Features: {top_n}")

        # 1. Prepare data from the entire historical record
        X_full, y_binary_full, y_continuous_full = self._prepare_full_dataset(category, timepoint, threshold_pct)

        # --- CORRECTED LOGIC: Select the most STABLE features across all seeds ---
        print(f"- Selecting features across {len(seeds)} seeds to find the most stable set...")
        all_features_list = []
        for seed in seeds:
            selected_features_for_seed = select_features_for_fold(X_full, y_binary_full, top_n, seed)
            all_features_list.extend(selected_features_for_seed)

        # Count the frequency of each feature
        feature_counts = pd.Series(all_features_list).value_counts()
        
        # Select the top_n most frequently occurring features
        final_selected_features = feature_counts.nlargest(top_n).index.tolist()
        
        print(f"- Identified the top {len(final_selected_features)} most stable features based on frequency.")
        print("Selected features:", final_selected_features)
        
        X_selected = X_full[final_selected_features].copy()

        # 3. Scale only the final, consolidated feature set
        continuous_features_selected = [f for f in X_selected.columns if X_selected[f].nunique() > 2]
        print(f"- Clipping and scaling {len(continuous_features_selected)} selected continuous features...")
        for col in continuous_features_selected:
            lower, upper = X_selected[col].quantile(0.01), X_selected[col].quantile(0.99)
            X_selected[col] = X_selected[col].clip(lower, upper)
        
        final_scaler = MinMaxScaler()
        X_selected[continuous_features_selected] = final_scaler.fit_transform(X_selected[continuous_features_selected])

        # 4. Get the optimal threshold from the previous validation run
        optimal_threshold = self._get_optimal_threshold(model_type, category, timepoint, threshold_pct, top_n)

        # 5. Save all common inference artifacts (scaler, features, threshold)
        strategy_dir_name = f"{model_type}_{category}_{timepoint}_{threshold_pct}pct"
        save_dir = os.path.join(self.models_dir, strategy_dir_name)
        os.makedirs(save_dir, exist_ok=True)
        
        joblib.dump(final_scaler, os.path.join(save_dir, 'final_scaler.joblib'))
        joblib.dump(final_selected_features, os.path.join(save_dir, 'final_features.joblib'))
        joblib.dump(optimal_threshold, os.path.join(save_dir, 'optimal_threshold.joblib'))
        print(f"- Saved all common artifacts to '{strategy_dir_name}'.")

        # 6. Train final models (one per seed) on the fully prepared data
        for seed in tqdm(seeds, desc="Training Final Models Across Seeds"):
            classifier, regressor = self._train_single_model_pair(X_selected, y_binary_full, y_continuous_full, seed, model_type)
            save_final_models(classifier, regressor, save_dir, seed)

        elapsed_time = timedelta(seconds=int(time.time() - start_time))
        print(f"\n### END ### Final models trained and saved in {elapsed_time}.")
        print(f"All artifacts are located in: {save_dir}")
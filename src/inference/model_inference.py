# In src/inference/model_inference.py

import os
import glob
import joblib
import pandas as pd
import numpy as np
import re

# Import helpers from the final model training utils for consistency
from src.training.utils.train_final_model_helpers import load_selected_features

class ModelInference:
    def __init__(self, model_type: str, category: str, timepoint: str, threshold_pct: int, top_n: int, optimize_for: str):
        # --- Parameters defining the single best strategy from backtesting ---
        self.model_type = model_type
        self.category = category
        self.timepoint = timepoint
        self.threshold_pct = threshold_pct
        self.top_n = top_n
        self.optimize_for = optimize_for
        
        # --- Define base directories ---
        self.base_dir = os.path.join(os.path.dirname(__file__), '../../data')
        self.final_models_dir = os.path.join(self.base_dir, 'models/final_deployment')
        self.stats_dir = os.path.join(self.base_dir, 'training/stats')
        self.output_dir = os.path.join(self.base_dir, 'inference')
        
        # --- Paths to specific files needed for inference ---
        self.feature_selection_file = os.path.join(self.base_dir, 'analysis/feature_selection/all_selected_features.xlsx')
        self.results_file = os.path.join(self.stats_dir, f"{self.model_type}_{self.category}_{self.optimize_for}_walk_forward_results.xlsx")

    def _load_final_models(self) -> dict:
        """Loads all final deployment models for the chosen strategy across all seeds."""
        models = {}
        strategy_dir_name = f"{self.model_type}_{self.category}_{self.timepoint}_{self.threshold_pct}pct_top{self.top_n}"
        model_dir = os.path.join(self.final_models_dir, strategy_dir_name)

        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f"Final model directory not found: {model_dir}")

        clf_files = glob.glob(os.path.join(model_dir, "final_clf_seed*.joblib"))
        reg_files = glob.glob(os.path.join(model_dir, "final_reg_seed*.joblib"))
        
        if not clf_files:
            raise FileNotFoundError(f"No final classifier models found in {model_dir}")
            
        for f in clf_files:
            seed = int(re.search(r'seed(\d+)', f).group(1))
            models[seed] = {'clf': joblib.load(f)}

        for f in reg_files:
            seed = int(re.search(r'seed(\d+)', f).group(1))
            if seed in models:
                models[seed]['reg'] = joblib.load(f)

        print(f"Loaded {len(models)} final model pairs from '{strategy_dir_name}'.")
        return models

    def _get_optimal_threshold(self) -> float:
        """Loads the pre-calculated optimal threshold 'X' from the walk-forward validation results."""
        results_df = pd.read_excel(self.results_file)
        
        strategy_results = results_df[
            (results_df['Timepoint'] == self.timepoint) &
            (results_df['Threshold'] == f"{self.threshold_pct}%") &
            (results_df['Top_n_Features'] == self.top_n)
        ]
        
        if strategy_results.empty:
            raise ValueError(f"Strategy {self.timepoint}-{self.threshold_pct}%-top{self.top_n} not found in results file.")
            
        optimal_threshold = strategy_results['Optimal Threshold'].mean()
        print(f"Loaded robust Optimal Threshold (X): {optimal_threshold:.4f}")
        return optimal_threshold

    def run(self, inference_df: pd.DataFrame):
        """
        Full inference pipeline for the chosen winning strategy.
        Accepts a dataframe of preprocessed features.
        """
        # --- 1. Load Models and the Pre-Determined Optimal Threshold ---
        models = self._load_final_models()
        optimal_threshold = self._get_optimal_threshold()
        
        # --- 2. Prepare Inference Data ---
        if inference_df is None or inference_df.empty:
            print("Inference data is empty. Nothing to predict.")
            return None
            
        strategy_name = f"{self.category}_{self.timepoint}_binary_{self.threshold_pct}pct"
        features = load_selected_features(self.feature_selection_file, self.category, strategy_name, self.top_n)
        X_inference = inference_df.reindex(columns=features, fill_value=0)

        # --- 3. Run Two-Stage Inference (Averaging Across Seeds) ---
        all_clf_probas, all_reg_preds = [], []
        for seed in sorted(models.keys()):
            model_pair = models[seed]
            all_clf_probas.append(model_pair['clf'].predict_proba(X_inference)[:, 1])
            if 'reg' in model_pair:
                all_reg_preds.append(model_pair['reg'].predict(X_inference))

        avg_clf_proba = np.mean(all_clf_probas, axis=0)
        avg_reg_pred = np.mean(all_reg_preds, axis=0) if all_reg_preds else np.zeros_like(avg_clf_proba)
        
        # --- 4. Generate Final Signals and Save Output ---
        output_df = inference_df[['Ticker', 'Filing Date']].copy()
        output_df['Classifier_Positive_Probability'] = avg_clf_proba
        output_df['Predicted_Return'] = avg_reg_pred
        
        output_df['Final_Signal'] = (
            (output_df['Classifier_Positive_Probability'] > 0.5) & 
            (output_df['Predicted_Return'] >= optimal_threshold)
        ).astype(int)

        os.makedirs(self.output_dir, exist_ok=True)
        strategy_name_for_file = f"{self.model_type}_{self.timepoint}_{self.threshold_pct}pct_top{self.top_n}"
        output_filename = f"inference_output_{strategy_name_for_file}.xlsx"
        output_path = os.path.join(self.output_dir, output_filename)
        
        output_df.sort_values(by='Final_Signal', ascending=False).to_excel(output_path, index=False)
        
        print(f"\nInference complete. {output_df['Final_Signal'].sum()} 'buy' signals generated.")
        print(f"Results saved to: {output_path}")

        return output_df

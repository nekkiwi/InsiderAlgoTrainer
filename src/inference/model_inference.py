import os
import joblib
import pandas as pd
from ast import literal_eval
from sklearn.base import is_classifier

from src.inference.utils.model_inference_helpers import load_feature_data

class ModelInference:
    def __init__(self):
        self.data_dir = os.path.join(os.path.dirname(__file__), '../../data')
        self.feature_selection_file = os.path.join(self.data_dir, "analysis/feature_selection/selected_features.xlsx")
        self.scraped_data_file = os.path.join(self.data_dir, "final/current_features_final.xlsx")
        self.output_dir = os.path.join(self.data_dir, "inference/")
        self.models = {}
        self.out_path = ""
        self.selected_features = []
        self.data = {}

    def load_models(self, model, target):
        # TODO: LIMSTOP IS NOT SUPPORTED
        """
        Load 5-fold models for the given target from model_dir.
        Expects files named '{target}_fold{i}.pkl' for i in 1..5.
        """
        model_dir = os.path.join(self.data_dir, f"models/{model}_{target}")
        for i in range(1, 6):
            path = os.path.join(model_dir, f"fold_{i}.joblib")
            self.models[f'fold_{i}'] = joblib.load(path)
        print(f"Loaded models for target '{target}': {list(self.models.keys())}")

    def load_feature_selection(self, target):
        """
        Read selected_features.xlsx, sheet named after target, cell C2 contains comma-separated features.
        """
        df = pd.read_excel(self.feature_selection_file, sheet_name=target)
        raw = df.iloc[0, 2]
        self.selected_features = list(literal_eval(raw))
        print(f"Selected features for '{target}': {self.selected_features}")

    def load_scraped_data(self):
        """
        Load the latest scraped data (e.g. top 20 insider buys) from Excel.
        """
        self.data = pd.read_excel(self.scraped_data_file)
        print(f"Loaded scraped data: {self.data.shape[0]} rows, {self.data.shape[1]} cols")

    def filter_features(self):
        """
        Subset self.data to only identifiers + selected features,
        aligning exactly to training.
        """
        id_cols = ['Ticker', 'Filing Date']
        # 1) move identifiers into index so reset_index won't re-introduce "index"
        df = self.data.set_index(id_cols)
        # 2) reindex to *only* selected_features, filling any missing ones with 0
        df = df.reindex(columns=self.selected_features, fill_value=0)
        # 3) bring identifiers back as columns
        self.data = df.reset_index()
        print(f"Filtered data to {len(self.selected_features)} features + identifiers: {self.data.shape}")



    def run_inference(self, target) -> pd.DataFrame:
        """
        Run inference using the loaded models on filtered data.
        Supports both classifiers (using predict_proba) and regressors (using predict).
        """
        X = self.data[self.selected_features]

        preds = []
        for name, model in self.models.items():
            if is_classifier(model):
                pred = model.predict_proba(X)[:, 1]
            else:
                pred = model.predict(X)
            preds.append(pred)

        avg_pred = sum(preds) / len(preds)

        result = self.data[['Ticker', 'Filing Date']].copy()
        result[f'{target}_score'] = avg_pred
        return result

    def save_output(self, result_df):
        os.makedirs(self.output_dir, exist_ok=True)
        result_df.to_excel(self.out_path, index=False)

    def run(self, features_df, models, targets, save_out):
        """
        Full pipeline: load models, features, data, filter, infer, save.
        """
        if features_df is None:
            self.data = load_feature_data(self.scraped_data_file)
        else:
            self.data = features_df
            
        for model in models:
            for target in targets: 
                self.load_models(model, target)
                self.load_feature_selection(target)
                self.filter_features()
                result_df = self.run_inference(target)
                if save_out: 
                    self.out_path = os.path.join(self.output_dir, f'{target}_inference_output.xlsx')
                    self.save_output(result_df)
                    print(f"Inference results saved to {self.out_path}")
        return result_df

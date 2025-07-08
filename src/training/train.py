import os
import time
from tqdm import tqdm
from datetime import timedelta
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_validate
from sklearn.metrics import accuracy_score, mean_squared_error, matthews_corrcoef, confusion_matrix
import joblib
from .utils.train_helpers import load_data, get_model, save_training_data

class ModelTrainer:
    def __init__(self):
        base = os.path.join(os.path.dirname(__file__), '../../data')
        self.features_file = os.path.join(base, 'final/features_final.xlsx')
        self.targets_file = os.path.join(base, 'final/targets_final.xlsx')
        self.sel_file = os.path.join(base, 'analysis/feature_selection/selected_features.xlsx')
        self.stats_dir = os.path.join(base, 'training/stats')
        self.preds_dir = os.path.join(base, 'training/predictions')
        self.models_dir = os.path.join(base, 'models')
        self.features_df = None
        self.targets = None
        self.train_idx = None
        self.test_idx = None

    def load_data(self):
        print("[INFO] Loading features and target datasets...")
        self.features_df, self.targets = load_data(self.features_file, self.targets_file)
        total = len(self.features_df)
        print(f"[INFO] Loaded {total} feature rows and {len(self.targets)} target sheets.")
        idx_all = list(self.features_df.index)
        self.train_idx, self.test_idx = train_test_split(
            idx_all, test_size=0.2, random_state=42, shuffle=True
        )
        print(f"[INFO] Defined train/test split: {len(self.train_idx)} train rows, {len(self.test_idx)} test rows.")

    def train_for_sheet(self, sheet, model_type, selected_features):
        print(f"[INFO] Starting training for sheet '{sheet}' with model '{model_type}'.")
        if sheet not in self.targets:
            print(f"[WARN] Sheet '{sheet}' not found, skipping.")
            return
        df_t = self.targets[sheet]
        is_reg = sheet.startswith('final_')
        records, preds = [], []
        print(f"[INFO] Processing {len(selected_features)} feature sets.")
        for row in selected_features.to_dict('records'):
            feats = [f.strip("'") for f in eval(row['Selected Features'])]
            print(f"[INFO] Limit={row.get('Limit')}, Stop={row.get('Stop')} | {len(feats)} features.")
            X = self.features_df.loc[:, feats]
            if is_reg:
                y = df_t[sheet]
            else:
                col = f"Limit {row['Limit']}, Stop {row['Stop']}"
                if col not in df_t.columns:
                    print(f"[WARN] Column '{col}' missing, skipping.")
                    continue
                y = df_t[col]
            common = X.index.intersection(y.dropna().index)
            print(f"[INFO] {len(common)} rows align with target.")
            train_idx = [i for i in common if i in self.train_idx]
            test_idx = [i for i in common if i in self.test_idx]
            print(f"[INFO] {len(train_idx)} train rows, {len(test_idx)} test rows.")
            if not train_idx or not test_idx:
                print("[WARN] Insufficient data for split, skipping.")
                continue
            Xtr, ytr = X.loc[train_idx], y.loc[train_idx]
            Xte, yte = X.loc[test_idx], y.loc[test_idx]
            cv = KFold(5, shuffle=True, random_state=42) if is_reg else StratifiedKFold(5)
            model = get_model(model_type, is_reg)
            print("[INFO] Running cross-validation folds...")
            cv_res = cross_validate(model, Xtr, ytr, cv=cv, return_estimator=True)
            for i, est in tqdm(enumerate(cv_res['estimator'], 1), desc="Saving CV folds", total=5):
                est.fit(Xtr, ytr)
                path = os.path.join(self.models_dir, f"{model_type}_{sheet}", f"fold_{i}.joblib")
                os.makedirs(os.path.dirname(path), exist_ok=True)
                joblib.dump(est, path)
            print("[INFO] Fitting final model...")
            model.fit(Xtr, ytr)
            ypred = model.predict(Xte)
            if is_reg:
                print("[INFO] Computing regression metrics...")
                bin_true = (yte > 0).astype(int)
                bin_pred = (ypred > 0).astype(int)
                mse = mean_squared_error(yte, ypred)
                acc = accuracy_score(bin_true, bin_pred)
                mcc = matthews_corrcoef(bin_true, bin_pred)
                cm = confusion_matrix(bin_true, bin_pred)
                ppv = cm[1,1] / max((cm[1,1]+cm[0,1]),1)
                npv = cm[0,0] / max((cm[0,0]+cm[1,0]),1)
                records.append({'Limit': row['Limit'], 'Stop': row['Stop'], 'MSE': mse,
                                'Accuracy': acc, 'MCC': mcc, 'PPV': ppv, 'NPV': npv})
            else:
                print("[INFO] Computing classification accuracy...")
                acc = accuracy_score(yte, ypred)
                records.append({'Limit': row['Limit'], 'Stop': row['Stop'], 'Accuracy': acc})
            df_pred = pd.DataFrame({f'Pred_{sheet}': ypred, f'GT_{sheet}': yte}, index=yte.index)
            if not df_pred.empty:
                print(f"[INFO] Recording {len(df_pred)} predictions.")
                preds.append({'Predictions': df_pred, 'Limit': row['Limit'], 'Stop': row['Stop']})
        print(f"[INFO] Saving results for '{sheet}'.")
        save_training_data(records, preds, self.stats_dir, self.preds_dir, model_type, sheet, self.features_df)

    def run(self, model_type, timepoints, metrics):
        print(f"[START] {model_type} across {len(timepoints)} timepoints and {len(metrics)} metrics.")
        self.load_data()
        overall_start = time.time()
        for metric in metrics:
            for tp in timepoints:
                sheet = f"final_{metric}_{tp}_raw"
                try:
                    selected_features = pd.read_excel(self.sel_file, sheet_name=sheet)
                except Exception:
                    print(f"[WARN] No features for {sheet}, skipping.")
                    continue
                print(f"\n>>> {sheet} with {len(selected_features)} feature sets >>>")
                t0 = time.time()
                self.train_for_sheet(sheet, model_type, selected_features)
                duration = timedelta(seconds=int(time.time()-t0))
                print(f"<<< Completed {sheet} in {duration} <<<")
        total = timedelta(seconds=int(time.time()-overall_start))
        print(f"[COMPLETE] All done in {total}.")

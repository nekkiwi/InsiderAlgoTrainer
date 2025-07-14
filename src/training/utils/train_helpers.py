# In src/training/utils/train_helpers.py
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
from sklearn.metrics import matthews_corrcoef

def annualize_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Calculates the annualized Sharpe ratio for a series of daily returns."""
    if returns.std() == 0 or len(returns) < 2:
        return np.nan
    
    excess_returns = returns - risk_free_rate / 252
    sharpe = np.mean(excess_returns) / np.std(excess_returns)
    return sharpe * np.sqrt(252)

def adjusted_sharpe_ratio(sharpe: float, num_signals: int, target_signals: int = 100) -> float:
    """Adjusts the Sharpe Ratio based on the number of signals."""
    if pd.isna(sharpe) or num_signals <= 0:
        return 0.0
    adjustment_factor = min(1.0, np.sqrt(num_signals / target_signals))
    return sharpe * adjustment_factor

def find_optimal_threshold(predicted_returns: pd.Series, actual_returns: pd.Series) -> dict:
    """Finds the regressor output threshold that maximizes a chosen metric on a validation set."""
    best_score = -np.inf
    best_threshold = np.nan
    
    for percentile in range(1, 100):
        threshold = np.percentile(predicted_returns, percentile)
        final_selection_returns = actual_returns[predicted_returns >= threshold]
        
        if len(final_selection_returns) < 10: continue

        sharpe = annualize_sharpe_ratio(final_selection_returns)
        adj_sharpe = adjusted_sharpe_ratio(sharpe, len(final_selection_returns))
        
        current_score = adj_sharpe
            
        if pd.notna(current_score) and current_score > best_score:
            best_score = current_score
            best_threshold = threshold
            
    return {'validation_score': best_score, 'optimal_threshold': best_threshold}

def load_data(features_file, targets_file):
    """Loads and prepares features and targets."""
    features_df = pd.read_excel(features_file)
    targets_dict_raw = pd.read_excel(targets_file, sheet_name=None)
    features_df['Filing Date'] = pd.to_datetime(features_df['Filing Date'], dayfirst=True)
    targets_dict = {}
    for sheet_name, df in targets_dict_raw.items():
        df['Filing Date'] = pd.to_datetime(df['Filing Date'], dayfirst=True)
        targets_dict[sheet_name] = df
    return features_df, targets_dict

def load_selected_features(sel_file, category, strategy_name, top_n):
    """Loads top N features for a strategy."""
    sheet_name = f"{category.title()}_Features"
    try:
        df = pd.read_excel(sel_file, sheet_name=sheet_name)
        if strategy_name not in df.columns: return []
        scores = df[['Feature', strategy_name]].copy().dropna(subset=[strategy_name])
        top = scores[scores[strategy_name] > 0].sort_values(by=strategy_name, ascending=False)
        return top['Feature'].head(top_n).tolist()
    except (FileNotFoundError, ValueError): return []

def evaluate_test_fold(classifier, regressor, optimal_threshold, X_ts, y_bin_ts, y_cont_ts) -> dict:
    """Evaluates a single test fold and returns a dictionary of metrics, including the composite score."""
    if X_ts.empty or optimal_threshold is np.nan: return None

    buy_signals = classifier.predict(X_ts)
    if buy_signals.sum() == 0: return None
    
    pos_class_idx = X_ts.index[buy_signals == 1]
    neg_class_idx = X_ts.index[buy_signals == 0]
    returns_pos_classifier = y_cont_ts.loc[pos_class_idx]
    returns_neg_classifier = y_cont_ts.loc[neg_class_idx]

    _, p_val_classifier = mannwhitneyu(returns_pos_classifier, returns_neg_classifier, alternative='greater')
    
    sharpe_classifier = annualize_sharpe_ratio(returns_pos_classifier)
    adj_sharpe_classifier = adjusted_sharpe_ratio(sharpe_classifier, len(returns_pos_classifier))
    
    if regressor is None or returns_pos_classifier.empty: return None
    
    predicted_returns = pd.Series(regressor.predict(X_ts.loc[pos_class_idx]), index=pos_class_idx)
    final_selection_idx = predicted_returns[predicted_returns >= optimal_threshold].index
    final_returns = y_cont_ts.loc[final_selection_idx]
    
    _, p_val_final = (np.nan, np.nan)
    if not final_returns.empty and not returns_neg_classifier.empty:
        _, p_val_final = mannwhitneyu(final_returns, returns_neg_classifier, alternative='greater')
    
    sharpe_final = annualize_sharpe_ratio(final_returns)
    adj_sharpe_final = adjusted_sharpe_ratio(sharpe_final, len(final_returns))

    metrics = {
        'Adjusted Sharpe (Final)': adj_sharpe_final,
        'Adjusted Sharpe (Classifier)': adj_sharpe_classifier,
        'Sharpe Ratio (Final)': sharpe_final,
        'Sharpe Ratio (Classifier)': sharpe_classifier,
        'P-Value (Final vs Neg)': p_val_final,
        'P-Value (Classifier vs Neg)': p_val_classifier,
        'Num Signals (Final)': len(final_returns),
        'Num Signals (Classifier)': len(pos_class_idx),
        'Optimal Threshold': optimal_threshold,
        'Median Return (Final Strategy)': final_returns.median(),
        'Median Return (Classifier)': returns_pos_classifier.median(),
        'Mean Return (Final Strategy)': final_returns.mean(),
        'Mean Return (Classifier)': returns_pos_classifier.mean(),
        'MCC (Classifier)': matthews_corrcoef(y_bin_ts, buy_signals)
    }
    return metrics

    
def save_strategy_results(results_df, stats_dir, model_name, category):
    """
    Saves strategy results, now including separate parameter columns for the
    classifier and the regressor in the output summary.
    """
    if results_df.empty:
        print("[INFO] Results DataFrame is empty. No file will be saved.")
        return

    os.makedirs(stats_dir, exist_ok=True)
    # The filename now includes the model name and tuning status
    output_path = os.path.join(stats_dir, f"{model_name}_{category}_walk_forward_results.xlsx")

    # --- UPDATED: Group by the separate classifier and regressor parameters ---
    group_cols = [
        'Timepoint', 'Threshold', 'Fold', 'Model'
    ]
    
    display_cols = [
        'Adjusted Sharpe (Final)',
        'Adjusted Sharpe (Classifier)',
        'Sharpe Ratio (Final)',
        'Sharpe Ratio (Classifier)',
        'P-Value (Final vs Neg)',
        'P-Value (Classifier vs Neg)',
        'Num Signals (Final)',
        'Num Signals (Classifier)',
        'Optimal Threshold',
        'Median Return (Final Strategy)',
        'Median Return (Classifier)',
        'Mean Return (Final Strategy)',
        'Mean Return (Classifier)',
        'MCC (Classifier)'
    ]
    
    valid_display_cols = [col for col in display_cols if col in results_df.columns]
    
    # Calculate mean and std grouped by the detailed model configuration
    mean_df = results_df.groupby(group_cols)[valid_display_cols].mean().reset_index()
    std_df = results_df.groupby(group_cols)[valid_display_cols].std().reset_index()

    mean_df.columns = [col if col in group_cols else f"{col} (Mean)" for col in mean_df.columns]
    std_df.columns = [col if col in group_cols else f"{col} (Std)" for col in std_df.columns]

    summary_df = pd.merge(mean_df, std_df, on=group_cols, how='left')
    summary_df.sort_values(by=['Timepoint', 'Threshold', 'Fold'], inplace=True)

    try:
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            summary_df.to_excel(writer, sheet_name='Per-Fold Summary', index=False)
            results_df.to_excel(writer, sheet_name='All Results (Raw)', index=False)
        print(f"\n- Walk-forward strategy results saved to {output_path}")
    except Exception as e:
        print(f"[ERROR] Failed to save Excel file: {e}")
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

def find_optimal_threshold(predicted_returns: pd.Series, actual_returns: pd.Series, optimize_for: str = 'adjusted_sharpe') -> dict:
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
        if optimize_for == 'sharpe':
            current_score = sharpe
            
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
    
    # --- The composite score is the primary metric ---
    predictive_power_score = adj_sharpe_final

    metrics = {
        'Predictive Power Score': predictive_power_score,
        'Adjusted Sharpe (Final)': adj_sharpe_final,
        'Adjusted Sharpe (Classifier)': adj_sharpe_classifier,
        'P-Value (Final vs Neg)': p_val_final,
        'P-Value (Classifier vs Neg)': p_val_classifier,
        'Num Signals (Final)': len(final_returns),
        'Num Signals (Classifier)': len(pos_class_idx),
        'Optimal Threshold': optimal_threshold,
        'Median Return (Final Strategy)': final_returns.median(),
        'Median Return (Classifier)': returns_pos_classifier.median(),
        'MCC (Classifier)': matthews_corrcoef(y_bin_ts, buy_signals)
    }
    return metrics

def _convert_timepoint_to_days(tp_string):
    """Safely converts a timepoint string like '1w', '2m' to an integer number of days."""
    if pd.isna(tp_string): return np.nan
    match = re.match(r'(\d+)([a-zA-Z])', str(tp_string))
    if not match: return np.nan
    value, unit = int(match.group(1)), match.group(2).lower()
    if unit == 'w': return value * 7
    if unit == 'm': return int(value * 21) # Using 21 as a common approximation
    return value

def plot_final_summary(results_df, output_dir, category, optimize_for):
    """Generates and saves a grid of boxplots with selective y-axis clipping."""
    if results_df.empty: return

    results_df['Timepoint_numeric'] = results_df['Timepoint'].apply(_convert_timepoint_to_days)
    results_df['Threshold_numeric'] = results_df['Threshold'].str.replace('%', '').astype(int)
    results_df = results_df.sort_values(['Timepoint_numeric', 'Threshold_numeric', 'Top_n_Features'])
    results_df['Strategy'] = results_df['Timepoint'] + '-' + results_df['Threshold']
    strategy_order = results_df['Strategy'].unique()
    
    # --- Define metrics to plot and which ones to clip ---
    metrics_to_plot = [
        'Predictive Power Score', 'Adjusted Sharpe (Final)', 'Adjusted Sharpe (Classifier)','Median Return (Final Strategy)','Median Return (Classifier)',
        'MCC (Classifier)','Num Signals (Final)', 'Num Signals (Classifier)','P-Value (Final vs Neg)', 'P-Value (Classifier vs Neg)'
    ]
    metrics_to_clip = ['Predictive Power Score', 'Median Return (Final Strategy)', 'Adjusted Sharpe (Final)']
    
    plot_df = results_df.copy()
    for metric in metrics_to_clip:
        if metric in plot_df.columns:
            lower_bound = plot_df[metric].quantile(0.01)
            upper_bound = plot_df[metric].quantile(0.99)
            plot_df[metric] = plot_df[metric].clip(lower=lower_bound, upper=upper_bound)
    
    fig, axs = plt.subplots(2, 5, figsize=(70, 20), tight_layout=True)
    fig.suptitle(f'Walk-Forward Validation Results (Optimized for {optimize_for.replace("_", " ").title()})', fontsize=24, y=1.03)
    
    for i, metric in enumerate(metrics_to_plot):
        if i >= len(axs.flatten()): break
        ax = axs.flatten()[i]
        
        sns.boxplot(x='Strategy', y=metric, data=plot_df, order=strategy_order, ax=ax, hue='Strategy', legend=False)
        ax.set_title(metric, fontsize=16)
        ax.set_xlabel(None)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
        
        if 'P-Value' in metric:
            ax.axhline(0.05, ls='--', color='red', zorder=10, label='p=0.05 Significance')
            if not ax.get_legend(): ax.legend(fontsize=12)
            
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'walk_forward_{category}_{optimize_for}_summary_plot.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"- Final summary plot saved.")
    
def save_strategy_results(results_df, stats_dir, model_type, category, optimize_for):
    """Saves final per-fold results."""
    if results_df.empty: return
    os.makedirs(stats_dir, exist_ok=True)
    results_df.to_excel(os.path.join(stats_dir, f"{model_type}_{category}_{optimize_for}_walk_forward_results.xlsx"), index=False)
    print(f"\n- Walk-forward strategy results saved.")

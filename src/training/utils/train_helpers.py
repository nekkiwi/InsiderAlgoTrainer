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

def _convert_timepoint_to_days(tp_string):
    """Safely converts a timepoint string like '1w', '2m' to an integer number of days."""
    if pd.isna(tp_string): return np.nan
    match = re.match(r'(\d+)([a-zA-Z])', str(tp_string))
    if not match: return np.nan
    value, unit = int(match.group(1)), match.group(2).lower()
    if unit == 'w': return value * 7
    if unit == 'm': return int(value * 21) # Using 21 as a common approximation
    return value

def plot_final_summary(results_df, output_dir, category):
    """Generates and saves a grid of boxplots with selective y-axis clipping."""
    if results_df.empty: return

    results_df['Timepoint_numeric'] = results_df['Timepoint'].apply(_convert_timepoint_to_days)
    results_df['Threshold_numeric'] = results_df['Threshold'].str.replace('%', '').astype(int)
    results_df = results_df.sort_values(['Timepoint_numeric', 'Threshold_numeric'])
    results_df['Strategy'] = results_df['Timepoint'] + '-' + results_df['Threshold']
    strategy_order = results_df['Strategy'].unique()
    
    # --- Define metrics to plot and which ones to clip ---
    metrics_to_plot = ['Adjusted Sharpe (Final)', 'Adjusted Sharpe (Classifier)','Median Return (Final Strategy)','Median Return (Classifier)',
        'MCC (Classifier)','Num Signals (Final)', 'Num Signals (Classifier)','P-Value (Final vs Neg)', 'P-Value (Classifier vs Neg)'
    ]
    metrics_to_clip = ['Median Return (Final Strategy)', 'Adjusted Sharpe (Final)']
    
    plot_df = results_df.copy()
    for metric in metrics_to_clip:
        if metric in plot_df.columns:
            lower_bound = plot_df[metric].quantile(0.01)
            upper_bound = plot_df[metric].quantile(0.99)
            plot_df[metric] = plot_df[metric].clip(lower=lower_bound, upper=upper_bound)
    
    fig, axs = plt.subplots(3, 3, figsize=(60, 30), tight_layout=True)
    fig.suptitle(f'Walk-Forward Validation Results', fontsize=24, y=1.03)
    
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
    plt.savefig(os.path.join(output_dir, f'walk_forward_{category}_summary_plot.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"- Final summary plot saved.")
    
def save_strategy_results(results_df, stats_dir, model_type, category):
    """
    Saves strategy results with a per-fold summary sheet listed first.
    The summary calculates the mean and standard deviation across seeds for each fold.
    """
    if results_df.empty:
        print("[INFO] Results DataFrame is empty. No file will be saved.")
        return

    os.makedirs(stats_dir, exist_ok=True)
    output_path = os.path.join(stats_dir, f"{model_type}_{category}_walk_forward_results.xlsx")

    # --- MODIFIED: Added 'Fold' to the grouping columns ---
    # This is the key change to create a per-fold summary.
    # Add model and params to the grouping for the per-fold summary
    group_cols = ['Timepoint', 'Threshold', 'Fold', 'Model', 'Best Params']
    
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

    # --- This logic now computes stats per fold ---
    print("[INFO] Computing per-fold summary statistics across seeds...")
    
    # Ensure display_cols exist in the DataFrame to avoid errors
    valid_display_cols = [col for col in display_cols if col in results_df.columns]
    
    # Calculate mean and standard deviation grouped by strategy AND fold
    mean_df = results_df.groupby(group_cols)[valid_display_cols].mean().reset_index()
    std_df = results_df.groupby(group_cols)[valid_display_cols].std().reset_index()

    # Add suffixes to the metric columns for clarity
    mean_df.columns = [col if col in group_cols else f"{col} (Mean)" for col in mean_df.columns]
    std_df.columns = [col if col in group_cols else f"{col} (Std)" for col in std_df.columns]

    # Merge the mean and std dataframes to create the final summary
    summary_df = pd.merge(mean_df, std_df, on=group_cols, how='left')
    
    # Sort the summary for better readability
    summary_df.sort_values(by=group_cols, inplace=True)

    # Write the results to an Excel file with two sheets
    try:
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            summary_df.to_excel(writer, sheet_name='Per-Fold Summary', index=False)
            results_df.to_excel(writer, sheet_name='All Results (Raw)', index=False)
        print(f"\n- Walk-forward strategy results saved to {output_path}")
    except Exception as e:
        print(f"[ERROR] Failed to save Excel file: {e}")

import matplotlib.colors as mcolors
def plot_performance_dashboard(results_df: pd.DataFrame, output_dir: str, category: str):
    """
    Generates and saves a four-panel heatmap dashboard to compare all strategies across all folds.
    The strategies are sorted chronologically on the y-axis.
    """
    if results_df.empty:
        print("[PLOT] Results DataFrame is empty. Skipping dashboard plot.")
        return

    # --- Nested helper function for sorting ---
    def _convert_timepoint_to_days(tp_string):
        """Safely converts a timepoint string like '1w', '2m' to an integer number of days."""
        if pd.isna(tp_string): return np.nan
        match = re.match(r'(\d+)([a-zA-Z])', str(tp_string))
        if not match: return np.nan
        value, unit = int(match.group(1)), match.group(2).lower()
        if unit == 'w': return value * 7
        if unit == 'm': return int(value * 30) # Using 30 for month average
        return value # Default for days

    # --- 1. Data Preparation ---
    print("[PLOT] Preparing data for dashboard...")
    
    # Create numeric columns for sorting and sort the DataFrame
    results_df['Timepoint_numeric'] = results_df['Timepoint'].apply(_convert_timepoint_to_days)
    results_df['Threshold_numeric'] = results_df['Threshold'].str.replace('%', '').astype(int)
    results_df.sort_values(['Timepoint_numeric', 'Threshold_numeric'], inplace=True)
    
    # Create the strategy identifier *after* sorting
    results_df['Strategy'] = results_df['Timepoint'] + '-' + results_df['Threshold'].astype(str)

    # --- FIX: Capture the correct chronological order of strategies ---
    strategy_order = results_df['Strategy'].unique().tolist()

    group_cols = ['Strategy', 'Fold']
    metrics = {
        'Adjusted Sharpe (Final)': ['mean', 'std'],
        'Median Return (Final Strategy)': 'mean',
        'P-Value (Final vs Neg)': 'mean'
    }
    summary_df = results_df.groupby(group_cols).agg(metrics)
    
    try:
        sharpe_mean_pivot = summary_df[('Adjusted Sharpe (Final)', 'mean')].unstack()
        sharpe_std_pivot = summary_df[('Adjusted Sharpe (Final)', 'std')].unstack()
        median_return_pivot = summary_df[('Median Return (Final Strategy)', 'mean')].unstack()
        p_value_pivot = summary_df[('P-Value (Final vs Neg)', 'mean')].unstack()
    except KeyError as e:
        print(f"[PLOT ERROR] A required column is missing for pivoting: {e}")
        return

    # --- FIX: Re-index the pivot tables to enforce the correct chronological sort order ---
    sharpe_mean_pivot = sharpe_mean_pivot.reindex(strategy_order)
    sharpe_std_pivot = sharpe_std_pivot.reindex(strategy_order)
    median_return_pivot = median_return_pivot.reindex(strategy_order)
    p_value_pivot = p_value_pivot.reindex(strategy_order)

    # --- 2. Colormap and Normalization Setup (Unchanged) ---
    cmap_perf = sns.color_palette("Greens", as_cmap=True)
    norm_perf = mcolors.Normalize(vmin=0, vmax=3)
    cmap_stab = sns.color_palette("Reds", as_cmap=True)
    norm_stab = mcolors.Normalize(vmin=0, vmax=3)
    cmap_ret = sns.color_palette("Greens", as_cmap=True)
    norm_ret = mcolors.Normalize(vmin=0, vmax=0.10)
    cmap_pval = mcolors.ListedColormap(['#1a9850', '#d9ef8b', '#fee08b', '#bababa'])
    norm_pval = mcolors.BoundaryNorm([0, 0.01, 0.05, 0.1, 1.0], cmap_pval.N)

    # --- 3. Plotting ---
    fig, axes = plt.subplots(1, 4, figsize=(32, max(12, len(sharpe_mean_pivot.index) * 0.5)), sharey=True)
    fig.suptitle(f'Strategy Performance Dashboard ({category})', fontsize=20, y=0.95)

    plot_params = [
        (sharpe_mean_pivot, 'Mean Adjusted Sharpe (Performance)', cmap_perf, norm_perf, axes[0]),
        (sharpe_std_pivot, 'Std Dev of Sharpe (Stability)', cmap_stab, norm_stab, axes[1]),
        (median_return_pivot, 'Median Return (Typical Outcome)', cmap_ret, norm_ret, axes[2]),
        (p_value_pivot, 'P-Value (Significance)', cmap_pval, norm_pval, axes[3])
    ]

    for pivot_df, title, cmap, norm, ax in plot_params:
        sns.heatmap(pivot_df, ax=ax, cmap=cmap, norm=norm, annot=True, fmt=".2f", linewidths=.5,
                    cbar_kws={"orientation": "horizontal", "pad": 0.05, "shrink": 0.7})
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Walk-Forward Fold', fontsize=12)
        ax.set_ylabel('Strategy' if ax == axes[0] else '', fontsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    
    # --- 4. Saving the file ---
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f'performance_dashboard_{category}.png')
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"- Performance dashboard saved to {file_path}")
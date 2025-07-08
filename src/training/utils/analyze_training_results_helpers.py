# helper_functions.py

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, matthews_corrcoef,
    precision_score, recall_score, f1_score,
    mean_squared_error
)
import os
import matplotlib.pyplot as plt
import seaborn as sns

import os
import re
from typing import List, Tuple

def get_combinations(data_dir: str = 'data') -> List[Tuple[str, str]]:
    """
    Scan data/training/predictions/{model}/ for every file named
      pred_{model}_{target}.xlsx
    and return a sorted list of (model, target) tuples.
    """
    base = os.path.join(data_dir, 'training', 'predictions')
    if not os.path.isdir(base):
        raise FileNotFoundError(f"No predictions folder found at {base!r}")

    pattern = re.compile(r'^pred_(?P<model>[^_]+)_(?P<target>.+)\.xlsx$', re.IGNORECASE)
    combos = set()

    # each subdirectory is a model
    for model in os.listdir(base):
        model_dir = os.path.join(base, model)
        if not os.path.isdir(model_dir):
            continue

        for fname in os.listdir(model_dir):
            if "final" in fname:
                m = pattern.match(fname)
                if not m:
                    continue
                # double-check the folder name matches the model group
                model_name = m.group('model')
                target     = m.group('target')
                if model_name.lower() != model.lower():
                    # you could warn here if you like
                    continue
                combos.add((model_name, target))

    return sorted(combos)


def load_combination_data(model: str,
                          target: str,
                          data_dir: str = 'data'
                         ) -> pd.DataFrame:
    """
    Load a single (model, target) prediction file and drop rows with missing GT or Pred.
    
    Expects the file at:
        {data_dir}/training/predictions/pred_{model}_{target}.xlsx
    """
    # build the full path in a cross-platform way
    file_name = f"pred_{model}_{target}.xlsx"
    path = os.path.join(data_dir, 'training', 'predictions', model, file_name)
    
    # sanity check
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Could not find prediction file: {path}")
    
    # force use of openpyxl (works in modern notebooks)
    try:
        df = pd.read_excel(path, engine='openpyxl')
    except ValueError:
        # fallback: read & concat all sheets if there’s more than one
        xls = pd.ExcelFile(path, engine='openpyxl')
        df = pd.concat(
            pd.read_excel(xls, sheet_name=sheet, engine='openpyxl')
            for sheet in xls.sheet_names
        )
    
    # 3) Helper to locate the right column
    def _find_col(prefix: str) -> str:
        exact = f"{prefix}_{target}"
        if exact in df.columns:
            return exact

        # match any column that starts with "{prefix}_{target}_..."
        pattern = f"{prefix}_{target}_"
        matches = [c for c in df.columns if c.startswith(pattern)]
        if len(matches) == 1:
            return matches[0]
        elif len(matches) > 1:
            raise KeyError(
                f"Ambiguous columns for {prefix!r}/{target!r}: {matches}"
            )
        else:
            return None

    gt_col   = _find_col('GT')
    pred_col = _find_col('Pred')

    if not gt_col or not pred_col:
        raise KeyError(
            "Could not locate your GT/Pred columns.\n"
            f"  Expected at least one of:\n"
            f"    GT_{target} or GT_{target}_*\n"
            f"    Pred_{target} or Pred_{target}_*\n"
            f"  Found columns: {list(df.columns)}"
        )

    # 4) Rename and clean
    df = df.rename(columns={gt_col: 'GT', pred_col: 'Pred'})
    df = df.dropna(subset=['GT', 'Pred']).reset_index(drop=True)
    return df

def compute_metrics(df, top_k=100):
    from sklearn.metrics import (
        mean_squared_error, accuracy_score, matthews_corrcoef, 
        precision_score, recall_score, f1_score
    )

    # Detect if GT is binary
    is_binary = set(df['GT'].unique()).issubset({0, 1})

    # Prep binary labels
    df['GT_bin'] = df['GT'] if is_binary else (df['GT'] > 0).astype(int)
    df['Pred_bin'] = df['Pred'] if is_binary else (df['Pred'] > 0).astype(int)

    # Classification metrics
    acc = accuracy_score(df['GT_bin'], df['Pred_bin'])
    mcc = matthews_corrcoef(df['GT_bin'], df['Pred_bin'])
    precision = precision_score(df['GT_bin'], df['Pred_bin'], zero_division=0)
    recall = recall_score(df['GT_bin'], df['Pred_bin'], zero_division=0)
    f1 = f1_score(df['GT_bin'], df['Pred_bin'], zero_division=0)

    # Top-K hit ratio (only makes sense for continuous GT)
    if not is_binary:
        top_df = df.nlargest(top_k, 'Pred')
        hit_ratio = np.mean(top_df['GT'] > 0)
    else:
        hit_ratio = np.nan

    metrics = {
        'Accuracy': acc,
        'MCC': mcc,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        f'Top-{top_k} Hit': hit_ratio,
    }

    if not is_binary:
        mse = mean_squared_error(df['GT'], df['Pred'])
        directional_acc = np.mean(np.sign(df['GT']) == np.sign(df['Pred']))
        residuals = df['Pred'] - df['GT']
        metrics.update({
            'MSE': mse,
            'Directional Acc': directional_acc,
            'Residual Mean': residuals.mean(),
            'Residual Std': residuals.std(),
            'Residual Skew': residuals.skew(),
        })

    return metrics



def load_results(combinations):
    all_results = []

    for model, target in combinations:
        try:
            df = load_combination_data(model, target)
            metrics = compute_metrics(df)
            metrics['Model'] = model
            metrics['Target'] = target
            all_results.append(metrics)
        except Exception as e:
            print(f"Error loading {model}-{target}: {e}")

    results_df = pd.DataFrame(all_results)
    results_df.set_index(['Model', 'Target'], inplace=True)
    return results_df



def plot_metrics(results_df):
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Select and transpose metric data
    metrics = ['Accuracy', 'MCC', 'Precision', 'Recall', 'F1']
    data = results_df[metrics].copy().T  # shape: (metrics, (model, target))

    # Normalize each row (metric) for color scaling only
    data_normalized = data.copy()
    for metric in data.index:
        row = data.loc[metric]
        data_normalized.loc[metric] = (row - row.min()) / (row.max() - row.min())

    # Plot a single clean heatmap
    plt.figure(figsize=(max(12, len(data.columns) * 0.6), 2 + len(metrics)))
    sns.heatmap(
        data_normalized,
        annot=data,
        fmt=".2f",
        cmap="YlGnBu",
        linewidths=0.5,
        cbar_kws={"label": "Row-normalized Color"}
    )

    plt.title("Metric Grid (Raw Values, Row-normalized Color)", fontsize=14)
    plt.ylabel("Metric")
    plt.xlabel("Model / Target")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
    
    
def calculate_thresholds(combinations):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from tqdm import tqdm

    xls = pd.ExcelFile('data/final/targets_final.xlsx')
    thresholds = np.round(np.arange(-0.02, 0.301, 0.01), 4)
    all_stats = []
    titles = []

    for model, target in tqdm(combinations, desc='Model/Target Loop'):
        if 'final' not in target:
            continue

        df_pred = load_combination_data(model, target)
        df_pred = df_pred[['Ticker', 'Filing Date', 'Pred']].dropna().copy()
        df_pred['Filing Date'] = pd.to_datetime(df_pred['Filing Date'], dayfirst=True)

        # Choose return sheet based on frequency in target
        if '1d' in target:
            return_sheet = 'final_return_1d_raw'
            alpha_sheet = 'final_alpha_1d_raw'
        elif '2d' in target:
            return_sheet = 'final_return_1d_raw'
            alpha_sheet = 'final_alpha_1d_raw'
        elif '1w' in target:
            return_sheet = 'final_return_1w_raw'
            alpha_sheet = 'final_alpha_1w_raw'
        elif '2w' in target:
            return_sheet = 'final_return_2w_raw'
            alpha_sheet = 'final_alpha_2w_raw'
        elif '3w' in target:
            return_sheet = 'final_return_3w_raw'
            alpha_sheet = 'final_alpha_3w_raw'
        elif '1m' in target:
            return_sheet = 'final_return_1m_raw'
            alpha_sheet = 'final_alpha_1m_raw'
        elif '6w' in target:
            return_sheet = 'final_return_6w_raw'
            alpha_sheet = 'final_alpha_6w_raw'
        elif '2m' in target:
            return_sheet = 'final_return_2m_raw'
            alpha_sheet = 'final_alpha_2m_raw'

        returns = pd.read_excel(xls, sheet_name=return_sheet, parse_dates=['Filing Date'])
        returns.columns = returns.columns.str.strip()
        ret_col = [c for c in returns.columns if c not in ['Ticker', 'Filing Date']][0]
        returns = returns.rename(columns={ret_col: 'Return'})

        alphas = pd.read_excel(xls, sheet_name=alpha_sheet, parse_dates=['Filing Date'])
        alphas.columns = alphas.columns.str.strip()
        alpha_col = [c for c in alphas.columns if c not in ['Ticker', 'Filing Date']][0]
        alphas = alphas.rename(columns={alpha_col: 'Alpha'})

        merged = pd.merge(
            df_pred,
            returns[['Ticker', 'Filing Date', 'Return']],
            on=['Ticker', 'Filing Date'], how='inner'
        )
        merged = pd.merge(
            merged,
            alphas[['Ticker', 'Filing Date', 'Alpha']],
            on=['Ticker', 'Filing Date'], how='inner'
        )

        stats = []
        for t in thresholds:
            sel = merged[merged['Pred'] >= t]
            n = len(sel)
            if n == 0:
                continue

            avg_ret = sel['Return'].mean()
            med_ret = sel['Return'].median()
            avg_alpha = sel['Alpha'].mean()
            med_alpha = sel['Alpha'].median()
            vol_ret = sel['Return'].std()
            sharpe = avg_ret / vol_ret if vol_ret and n > 1 else np.nan
            hit_rate = (sel['Return'] > 0).mean()

            stats.append({
                'Model': model,
                'Target': target,
                'Threshold': t,
                'Num Investments': n,
                'Avg Return': avg_ret,
                'Median Return': med_ret,
                'Avg Alpha': avg_alpha,
                'Median Alpha': med_alpha,
                'Volatility': vol_ret,
                'Sharpe Ratio': sharpe,
                'Hit Rate': hit_rate
            })

        df_stats = pd.DataFrame(stats)
        all_stats.append(df_stats)
        titles.append((model, target))

    summary_df = pd.concat(all_stats, ignore_index=True)

    for (model, target), df_stats in zip(titles, all_stats):
        fig, axs = plt.subplots(1, 5, figsize=(25, 5))
        fig.suptitle(f"{model} / {target}", fontsize=16)

        axs[0].plot(df_stats['Threshold'], df_stats['Avg Return'], marker='o', label='Return')
        axs[0].plot(df_stats['Threshold'], df_stats['Avg Alpha'], marker='x', label='Alpha')
        axs[0].set_title('Avg Return & Alpha')
        axs[0].legend()

        axs[1].plot(df_stats['Threshold'], df_stats['Median Return'], marker='o', label='Return')
        axs[1].plot(df_stats['Threshold'], df_stats['Median Alpha'], marker='x', label='Alpha')
        axs[1].set_title('Median Return & Alpha')
        axs[1].legend()

        axs[2].plot(df_stats['Threshold'], df_stats['Sharpe Ratio'], marker='o')
        axs[2].set_title('Sharpe Ratio')

        axs[3].plot(df_stats['Threshold'], df_stats['Hit Rate'], marker='o')
        axs[3].set_title('Hit Rate')
        
        axs[4].plot(df_stats['Threshold'], df_stats['Num Investments'], marker='o')
        axs[4].set_title('Num Investments')

        for ax in axs:
            ax.set_xlabel('Threshold')
            ax.grid(True)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
        
    # 2) make one big DataFrame
    summary_df = pd.concat(all_stats, ignore_index=True)

    # 3) filter to only those thresholds
    summary_small = summary_df[summary_df['Threshold'].isin(thresholds)]

    # 4) list of metrics to tabulate
    metrics = [
        'Avg Return',
        'Median Return',
        'Avg Alpha',
        'Median Alpha',
        'Sharpe Ratio',
        'Hit Rate',
        'Num Investments'
    ]

    # 5) for each metric, pivot and display
    # Specify your output path
    output_path = 'data/training/threshold_metrics.xlsx'
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        for metric in metrics:
            pivot = summary_small.pivot_table(
                index=['Model', 'Target'],
                columns='Threshold',
                values=metric
            ).sort_index(axis=1).round(4)
            # Excel sheet names have a 31-char limit, so we truncate if necessary
            sheet_name = metric if len(metric) <= 31 else metric[:28] + '...'
            pivot.to_excel(writer, sheet_name=sheet_name, merge_cells=False,)
    return summary_df


def print_best_thresholds(summary_df, inv_per_week = 7):
    # Set your minimum‐investments filter
    min_investments = 12*4*inv_per_week # on once a day on average (not really because lots of data is lost so way less probably) (data is over 5 years)

    # Filter out any rows with too few picks
    filtered = summary_df[summary_df['Num Investments'] > min_investments]

    # Metrics you want the top‐3 for
    metrics = ['Avg Return', 'Median Return', 'Avg Alpha', 'Median Alpha']

    for metric in metrics:
        print(f"\nTop 3 by {metric} (Num Investments > {min_investments}):")
        
        # pick the top 3 rows
        top3 = (
            filtered
            .nlargest(3, metric)
            .loc[:, ['Model','Target','Threshold','Num Investments', metric]]
        )
        
        # print nicely
        print(top3.to_string(index=False))
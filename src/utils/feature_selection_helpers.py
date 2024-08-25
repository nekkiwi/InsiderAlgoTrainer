import pandas as pd
from sklearn.feature_selection import SelectFpr, f_classif, chi2, f_regression, RFECV
from sklearn.model_selection import StratifiedKFold, KFold
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def calculate_anova(features, target):
    """Perform ANOVA F-test between categorical targets and continuous features."""
    selector = SelectFpr(score_func=f_classif, alpha=0.05)
    selector.fit(features, target)
    return selector

def calculate_chi_square(features, target):
    """Perform Chi-Square test between categorical targets and categorical features."""
    selector = SelectFpr(score_func=chi2, alpha=0.05)
    selector.fit(features, target)
    return selector

def calculate_pearson(features, target):
    """Perform Pearson correlation (f_regression) between continuous targets and continuous features."""
    selector = SelectFpr(score_func=f_regression, alpha=0.05)
    selector.fit(features, target)
    return selector

def calculate_spearman(features, target):
    """Perform Spearman correlation (approximated by f_regression) between continuous targets and categorical features."""
    # Approximate Spearman using rank-transformed data
    features_ranked = features.rank().astype(float)
    selector = SelectFpr(score_func=f_regression, alpha=0.05)
    selector.fit(features_ranked, target)
    return selector

def rfecv_selection(estimator, features, target, cv=5):
    """Apply Recursive Feature Elimination with Cross-Validation (RFECV) to select features."""
    if isinstance(target, (pd.Series, pd.DataFrame)) and target.nunique() > 2:
        cv_strategy = KFold(n_splits=cv)
    else:
        cv_strategy = StratifiedKFold(n_splits=cv)
    
    selector = RFECV(estimator, step=1, cv=cv_strategy, scoring='accuracy')
    selector.fit(features, target)
    return selector

def plot_selected_features(feature_importances, feature_names, output_dir, title):
    """Plot the selected feature importances."""
    sorted_idx = np.argsort(feature_importances)
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(sorted_idx)), feature_importances[sorted_idx], align='center')
    plt.yticks(range(len(sorted_idx)), np.array(feature_names)[sorted_idx])
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_dir)
    plt.close()

def compare_feature_lists(selection_results, method_names, output_dir):
    """Compare the selected feature lists across different methods and save the comparison."""
    comparison_df = pd.DataFrame(selection_results, index=method_names).T
    comparison_df.to_excel(output_dir)
    sns.heatmap(comparison_df.isna(), cbar=False)
    plt.title('Feature Selection Comparison')
    plt.tight_layout()
    plt.savefig(output_dir.replace('.xlsx', '.png'))
    plt.close()
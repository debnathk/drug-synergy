import pandas as pd
import numpy as np
from tabulate import tabulate

import pickle
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data/raw"
RAW_DATA_PATH = ROOT / "data/raw"
EMBEDDINGS_PATH = ROOT / "data/embeddings"

# Load your dataset
with open(RAW_DATA_PATH / "train_split.pkl", 'rb') as file:
    train_df = pickle.load(file)

with open(RAW_DATA_PATH / "val_split.pkl", 'rb') as file:
    val_df = pickle.load(file)


def compute_regression_metrics(y_true, y_pred):
    """
    Compute RMSE, SCC (Spearman), and PCC (Pearson) for regression.

    Args:
        y_true: Ground truth (1D array-like).
        y_pred: Predictions (1D array-like).

    Returns:
        dict with keys 'RMSE', 'SCC', 'PCC'.
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    pcc, _ = stats.pearsonr(y_true, y_pred)
    scc, _ = stats.spearmanr(y_true, y_pred)
    return {"RMSE": rmse, "SCC": float(scc), "PCC": float(pcc)}


def compute_metrics_with_std(y_true, y_pred, n_bootstrap=200, seed=None):
    """
    Compute RMSE, SCC, PCC and their standard deviations via bootstrap.

    Args:
        y_true: Ground truth (1D array-like).
        y_pred: Predictions (1D array-like).
        n_bootstrap: Number of bootstrap samples.
        seed: Random seed for reproducibility.

    Returns:
        dict with keys 'RMSE', 'SCC', 'PCC'; each value is (mean, std).
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    n = len(y_true)
    if n < 2:
        m = compute_regression_metrics(y_true, y_pred)
        return {k: (v, 0.0) for k, v in m.items()}
    rmse_list, scc_list, pcc_list = [], [], []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt, yp = y_true[idx], y_pred[idx]
        rmse_list.append(np.sqrt(mean_squared_error(yt, yp)))
        try:
            pcc_list.append(stats.pearsonr(yt, yp)[0])
            scc_list.append(stats.spearmanr(yt, yp)[0])
        except Exception:
            pcc_list.append(np.nan)
            scc_list.append(np.nan)
    return {
        "RMSE": (float(np.nanmean(rmse_list)), float(np.nanstd(rmse_list))),
        "SCC": (float(np.nanmean(scc_list)), float(np.nanstd(scc_list))),
        "PCC": (float(np.nanmean(pcc_list)), float(np.nanstd(pcc_list))),
    }


def format_metrics_table_pretty(metrics_with_std, title="Validation metrics"):
    """
    Format RMSE, SCC, PCC with mean (std) as a PrettyTable string.

    Args:
        metrics_with_std: dict from compute_metrics_with_std, keys RMSE, SCC, PCC,
                          values (mean, std).
        title: Optional table title.

    Returns:
        Formatted string (PrettyTable if available, else plain text table).
    """
    try:
        from prettytable import PrettyTable
        pt = PrettyTable(["Metric", "Value"])
        for name in ("RMSE", "SCC", "PCC"):
            mean, std = metrics_with_std.get(name, (np.nan, 0.0))
            pt.add_row([name, f"{mean:.2f} ({std:.2f})"])
        return f"\n{title}\n{pt.get_string()}\n"
    except ImportError:
        lines = [f"\n{title}", "-" * 30]
        for name in ("RMSE", "SCC", "PCC"):
            mean, std = metrics_with_std.get(name, (np.nan, 0.0))
            lines.append(f"  {name:<8} {mean:.2f} ({std:.2f})")
        return "\n".join(lines) + "\n"


def standardize(target_col):
    '''
    Standardize distribution using Scikit-learn
    '''
    # Transform target column
    scaler = StandardScaler()

    # Reshape target_col to 2D if it's 1D
    target_col = scaler.fit_transform(target_col.values.reshape(-1, 1))

    return target_col

def visualize_omics_data():
    # Print the omics distribution for cell-lines
    val_data = pd.read_pickle(str(DATA_PATH / "val_split.pkl"))
    val_data_head = val_data.loc[:4, :]
    # print(tabulate(val_data_head.values.tolist(), headers=val_data_head.columns.tolist(), tablefmt="pretty"))

    sample_val_data = val_data.iloc[0]
    # print(sample_val_data)
    sample_val_data_all_omics = sample_val_data['CellLine']
    sample_val_data_mRNA = sample_val_data_all_omics[0]
    sample_val_data_proteomics = sample_val_data_all_omics[1]
    sample_val_data_miRNA = sample_val_data_all_omics[2]


    # Create individual DataFrames for each omics type
    omics_data_mRNA = pd.DataFrame({'mRNA': sample_val_data_mRNA})
    omics_data_proteomics = pd.DataFrame({'Proteomics': sample_val_data_proteomics})
    omics_data_miRNA = pd.DataFrame({'miRNA': sample_val_data_miRNA})

    # Plot the distribution of mRNA data points
    plt.figure(figsize=(8, 4))
    sns.boxplot(data=omics_data_mRNA)
    plt.title('Distribution of mRNA Data Points')
    plt.ylabel('Values')
    plt.xlabel('mRNA')
    output_path_mRNA = ROOT / "assets/mRNA_distribution.png"
    plt.savefig(output_path_mRNA)
    plt.close()

    # Plot the distribution of Proteomics data points
    plt.figure(figsize=(8, 4))
    sns.boxplot(data=omics_data_proteomics)
    plt.title('Distribution of Proteomics Data Points')
    plt.ylabel('Values')
    plt.xlabel('Proteomics')
    output_path_proteomics = ROOT / "assets/proteomics_distribution.png"
    plt.savefig(output_path_proteomics)
    plt.close()

    # Plot the distribution of miRNA data points
    plt.figure(figsize=(8, 4))
    sns.boxplot(data=omics_data_miRNA)
    plt.title('Distribution of miRNA Data Points')
    plt.ylabel('Values')
    plt.xlabel('miRNA')
    output_path_miRNA = ROOT / "assets/miRNA_distribution.png"
    plt.savefig(output_path_miRNA)
    plt.close()

def visualize_target_col_distribution():
    # print(len(train_df['Synergy_ZIP']))

    # Plot the distribution of the 'Synergy_ZIP' column in train_df
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=train_df['Synergy_ZIP'])
    plt.title('Distribution of Synergy_ZIP in Training Data')
    plt.xlabel('Synergy_ZIP')
    output_path_zip = ROOT / "assets/synergy_zip_distribution (standardized).png"
    plt.savefig(output_path_zip)
    plt.close()

if __name__ == "__main__":
    std_target_col = standardize(train_df['Synergy_ZIP'])
    print(std_target_col.shape)





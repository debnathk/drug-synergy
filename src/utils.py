import pandas as pd
import numpy as np
from tabulate import tabulate

import ast
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data/raw"

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

if __name__ == "__main__":
    visualize_omics_data()





import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

from main.evaluation.statistical_analysis import calc_CD, calc_Ff, calc_Xf2, qa, ranking_model

PROJECT_ROOT = Path(__file__).resolve().parents[2]

parser = argparse.ArgumentParser(description="Evaluating models")
parser.add_argument("-f", "--results_folder", type=str, help="Results folder.", required=True)
parser.add_argument("-e", "--experiment", type=str, help="Analyzed experiment.", required=True)
args = parser.parse_args()
results_folder = args.results_folder
table_choice = args.experiment

base_folder = PROJECT_ROOT / f"{results_folder}/{table_choice}"
if table_choice == "ablation_study":
    models = ["params_q_hk", "params_q_hktanh", "params_qd_hk", "params_qd_hktanh"]
elif table_choice == "gg_comp":
    models = ["chipclass", "ggrbficann", "gmmgg", "ggrbf"]
elif table_choice == "gg_baselines" or table_choice == "multiclass":
    models = [
        "knn",
        "svm",
        "lightgbm",
        "random_forest",
        "xgboost",
        "resnet",
        "ggrbf",
    ]
else:
    raise ValueError(f"{table_choice} is invalid.")

dfs = []
for model in models:
    folder_path = f"{base_folder}/{model}"
    datasets = os.listdir(folder_path)

    datasets_vec = []
    result_vec = []
    for dataset in datasets:
        with open(f"{folder_path}/{dataset}", encoding="utf-8") as f:
            file_content = f.read().splitlines()
        dataset_name = file_content[0].split("&")[0].strip().split()[0]
        dataset_value = np.round(float(file_content[0].split("&")[1].strip().split()[0]), 4)
        datasets_vec.append(dataset_name)
        result_vec.append(dataset_value)

    df = pd.DataFrame({"dataset": datasets_vec, f"{model}": result_vec})
    df.sort_values("dataset", inplace=True)
    df[model] = df[model].astype(float)
    dfs.append(df)

merged_df = dfs[0]  # Start with the first dataframe in the list

# Loop through the rest of the dataframes and merge them one by one
for df in dfs[1:]:
    merged_df = pd.merge(merged_df, df, on="dataset", how="inner")

df_without_a = merged_df.drop("dataset", axis=1)

df_ranked = ranking_model(df_without_a)
avg_ranks = np.round(np.mean(df_ranked, axis=0).values.astype(float), 4)
avg_ranks = avg_ranks.astype(str)
new_row = np.insert(avg_ranks, 0, "avg rank")

merged_df.loc[len(merged_df)] = new_row
merged_df = merged_df.astype(str)

# Convert the DataFrame to a LaTeX table
latex_table = merged_df.to_latex(index=False, escape=False)
# Print or save the LaTeX table
print(latex_table)

L = df_ranked.shape[1]
N = df_ranked.shape[0]
numerator = L - 1
denominator = (L - 1) * (N - 1)

F = scipy.stats.f.ppf(q=0.95, dfn=L - 1, dfd=(L - 1) * (N - 1))
print(f"F({numerator},{denominator})", F)

N = df_ranked.shape[0]
k = df_ranked.shape[1]
R = avg_ranks.astype(float)
print(f"Number of datasets: {N}")
print(f"Number of algorithms: {k}")
print(f"Numerator: {numerator}")
print(f"Denominator: {denominator}")
Ff = calc_Ff(N, k, calc_Xf2(N, k, R))
print(f"Ff: {Ff}")

qa_value = qa[str(L)]

print("qa", qa_value)
print("CD", np.round(calc_CD(L, N, qa_value), 4))

abs_diff = np.abs(R - np.min(R))

print(abs_diff)

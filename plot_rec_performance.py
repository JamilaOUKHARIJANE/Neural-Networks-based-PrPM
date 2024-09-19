import pandas as pd
import matplotlib.pyplot as plt
import os

from src.commons import log_utils, shared_variables as shared

import pdb
import csv
import numpy as np
from statistics import mean


def aggregate_results(log_name, alg,models_folder, beam_size, resource=False,timestamp=False,outcome=False,probability_reduction=False ):
    df = pd.DataFrame()
    for fold in range(shared.folds):
        eval_algorithm = alg + "_cf" + "r"*resource + "t"*timestamp + "o"*outcome
        folder_path = shared.output_folder / models_folder / str(fold) / 'results' / eval_algorithm
        print(f"fold {fold} - {eval_algorithm}")
        log_path = shared.log_folder / log_name
        log_data = log_utils.LogData(log_path)
        if alg == "beamsearch":
            filename = f'{log_data.log_name.value}_beam{str(beam_size)}_fold{str(fold)}_cluster{log_data.evaluation_prefix_start}{"_probability_reduction" * probability_reduction}.csv'
        else:
            filename = f'{log_data.log_name.value}_{str(alg)}_fold{str(fold)}_cluster{log_data.evaluation_prefix_start}.csv'

        df_results = pd.read_csv(os.path.join(folder_path, filename))
        grouped_df = df_results.groupby(['Prefix length'], as_index=False).agg(
            {'Damerau-Levenshtein Acts': 'mean', 'Damerau-Levenshtein Resources': 'mean'})
        df = pd.concat([df, grouped_df])
    grouped_df = df_results.groupby(['Prefix length'], as_index=False).agg(
        {'Damerau-Levenshtein Acts': 'mean', 'Damerau-Levenshtein Resources': 'mean'})
    return grouped_df

def add_plot(log, alg, models_folder, beam_size, encoding, id_col, id_row,probability_reduction=False):
    df = pd.DataFrame()
    df = pd.concat([df, aggregate_results(log, alg, models_folder, beam_size, resource=True, timestamp=False,
                                          outcome=False)])
    res_color = {
        'LSTM_One_hot': 'red' if probability_reduction else 'blue',
        'LSTM_Simple_categorical': 'orange' if probability_reduction else 'purple',
        'keras_trans_One_hot': 'green' if probability_reduction else 'brown',
        'keras_trans_Simple_categorical': 'gray' if probability_reduction else 'pink',
        'LSTM_Combined_Act_res': 'black' if probability_reduction else 'salmon',
        'keras_trans_Combined_Act_res': 'darkviolet' if probability_reduction else 'mediumpurple'
    }

    act_color = {
        'LSTM_One_hot': 'lime' if probability_reduction else 'crimson',
        'LSTM_Simple_categorical': 'yellow' if probability_reduction else 'coral',
        'keras_trans_One_hot': 'deepskyblue' if probability_reduction else 'teal',
        'keras_trans_Simple_categorical': 'royalblue' if probability_reduction else 'lightgreen',
        'LSTM_Combined_Act_res': 'tomato' if probability_reduction else 'yellowgreen',
        'keras_trans_Combined_Act_res': 'lightgray' if probability_reduction else 'tan'
    }

    axs[id_row][id_col].plot(df['Prefix length'],
                             df['Damerau-Levenshtein Acts'],
                             color=act_color[models_folder],
                             marker='+' if probability_reduction else '1',
                             label=models_folder+"_proba_reduction"* probability_reduction +" (Acts)")
    axs[id_row][id_col].plot(df['Prefix length'],
                             df['Damerau-Levenshtein Resources'],
                             color=res_color[models_folder],
                             marker='*' if probability_reduction else 'x',
                             label=models_folder+"_proba_reduction"* probability_reduction +" (Resources)")

if __name__ == "__main__":
    # Compute and save aggregated results
    nrows = len(shared.log_list)
    ncols = 3
    fig_width_px = 8
    f, axs = plt.subplots(nrows, ncols, figsize=(fig_width_px * ncols, fig_width_px * nrows))
    for log_id, log in enumerate(shared.log_list):
        for alg_id, alg in enumerate(["baseline", "beamsearch (beam_size=3)", "beamsearch (beam_size=5)"]):
            id_col = alg_id
            axs[log_id][id_col].set_xlabel('Prefix length')
            axs[log_id][id_col].set_ylabel('Avg. Damerau-Levenshtein Distance')
            axs[log_id][id_col].set_title(log.removesuffix(".csv") + "(" + alg + ")")
            for models_folder in ["LSTM","keras_trans"]:
                for encoder in ["_One_hot", "_Simple_categorical", "_Combined_Act_res"]:
                    if (alg== "beamsearch (beam_size=3)"):
                        add_plot(log, "beamsearch", models_folder + encoder, 3, encoder, id_col, log_id,
                                 probability_reduction=False)
                        add_plot(log, "beamsearch", models_folder + encoder, 3, encoder, id_col, log_id,
                                 probability_reduction=True)
                    elif alg== "beamsearch (beam_size=5)":
                        add_plot(log, "beamsearch", models_folder + encoder, 5, encoder, id_col, log_id,
                                 probability_reduction=False)
                        add_plot(log, "beamsearch", models_folder + encoder, 5, encoder, id_col, log_id,
                                 probability_reduction=True)
                    else:
                        add_plot(log, alg, models_folder + encoder, 0, encoder, id_col, log_id)
            axs[log_id][id_col].legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol = 2, fontsize = 'small')
    # Adjust the layout to make space for legends
    plt.tight_layout()
    title = "aggregated_distances_per_prefix"
    plt.savefig(os.path.join(shared.output_folder, f'{title}.pdf'))



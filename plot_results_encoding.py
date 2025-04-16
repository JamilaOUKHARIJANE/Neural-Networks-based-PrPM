from enum import Enum

import pandas as pd
import matplotlib.pyplot as plt
import os

from pathlib import Path

import pm4py

from results_aggregator_fold import set_log_ths
from src.commons import log_utils, shared_variables as shared

class LogName(Enum):
    SEPSIS = 'Sepsis_cases'
    HELPDESK = 'helpdesk'
    BPIC13 = 'BPI2013'
    BPIC20_1 = "PrepaidTravelCost"
    BPIC20_2 = "PermitLog"
    BPIC20_3 = "RequestForPayment"
    BPIC20_4 = "DomesticDeclarations"
    BPIC20_5 = "InternationalDeclarations"

def getprefix_start(log_path):
    log_name = log_path.stem
    if log_name == LogName.HELPDESK.value:
        median = 5
        evaluation_prefix_start = median // 2 - 1
    elif log_name == LogName.BPIC13.value:
        median = 4
        evaluation_prefix_start = median // 2 - 1
    elif log_name == LogName.SEPSIS.value:
        median = 13
        evaluation_prefix_start = median // 2 - 2
    elif log_name == LogName.BPIC20_1.value:
        median = 8
        evaluation_prefix_start = median // 2 - 2
    elif log_name == LogName.BPIC20_2.value:
        median = 11
        evaluation_prefix_start = median // 2 - 2
    elif log_name == LogName.BPIC20_3.value:
        median = 5
        evaluation_prefix_start = median // 2 - 1
    elif log_name == LogName.BPIC20_4.value:
        median = 5
        evaluation_prefix_start = median // 2 - 1
        weight = 0.9
    elif log_name == LogName.BPIC20_5.value:
        median = 10
        evaluation_prefix_start = median // 2 - 2
    else:
        raise RuntimeError(f"No settings defined for log: {log_name.value}.")
    return evaluation_prefix_start

def aggregate_results(log_path,alg,encoder,models_folder="keras_trans", beam_size=0, resource=False,timestamp=False,outcome=False,BK=False, BK_end=False ):
    df = pd.DataFrame()
    log_name = log_path.stem
    if alg=='SUTRAN':
        folder_path = shared.output_folder / 'SUTRAN_results'
        filename = f'{log_name}_results.csv'
        file_path = os.path.join(folder_path, filename)
        if not os.path.exists(file_path):
            return 0, 0, 0, 0
        df_results = pd.read_csv(file_path, delimiter=',')
        df_results = df_results.rename(columns={'dam_lev_similarity': 'Damerau-Levenshtein Acts',
                                                'dam_lev_res_similarity': 'Damerau-Levenshtein Resources',
                                                'pref_len': 'Prefix length'})
        grouped_df = df_results.groupby(['Prefix length'], as_index=False).agg(
            {'Damerau-Levenshtein Acts': 'mean', 'Damerau-Levenshtein Resources': 'mean'})
        return grouped_df
    models_folder = models_folder + encoder
    for fold in range(shared.folds):
        eval_algorithm = alg + "_cf" + "r"*resource + "t"*timestamp + "o"*outcome
        if log_name in ['helpdesk', 'BPI2013'] and not BK_end:
            folder_path = shared.output_folder / models_folder / str(fold) / 'results' / eval_algorithm
        else:
            folder_path = shared.output_folder / models_folder / str(fold) / 'results_new' / eval_algorithm
        evaluation_prefix_start,traces_id,weight = set_log_ths(log_path,encoder,beam_size)
        if alg == "beamsearch":
            filename = f'{log_name}_beam{str(beam_size)}_fold{str(fold)}_cluster{evaluation_prefix_start}{"_mean_BK" * BK}{"_BKatEND" * BK_end}.csv'
        else:
            filename = f'{log_name}_{str(alg)}_fold{str(fold)}_cluster{evaluation_prefix_start}.csv'
        df_results = pd.read_csv(os.path.join(folder_path, filename))
        if BK and log_name not in ['BPI2013', 'helpdesk']:
            df_results = df_results[df_results['Weight'] == weight]
        if log_name == 'helpdesk':
            df_results = df_results[df_results['Prefix length'] < 5]
        df_results = df_results[df_results['Case ID'].isin(traces_id)] if alg == "beamsearch" else df_results[
            df_results['Case Id'].isin(traces_id)]
        grouped_df = df_results.groupby(['Prefix length'], as_index=False).agg(
            {'Damerau-Levenshtein Acts': 'mean', 'Damerau-Levenshtein Resources': 'mean'})
        df = pd.concat([df, grouped_df])
    grouped_df = df.groupby(['Prefix length'], as_index=False).agg(
        {'Damerau-Levenshtein Acts': 'mean', 'Damerau-Levenshtein Resources': 'mean'})
    return grouped_df

def add_plot(axs, dataset, metric, results,encoder):
    handles = []
    labels = []
    for i in results.keys():
        result_list = results[i][metric]
        prefix_length_list = results[i]["Prefix length"]
        line, =axs.plot(prefix_length_list, result_list,
                 color=shared.method_color[i],
                 marker=shared.method_marker[i],
                 label=i)
        handles.append(line)
        labels.append(i)
    return handles, labels

def plot_results(dataset_results,prefix_length):
    for metric in ["Damerau-Levenshtein Acts"]:#, "Damerau-Levenshtein Resources"]:
        folder_path = Path.cwd() / 'evaluation plots' / metric
        if not Path.exists(folder_path):
            Path.mkdir(folder_path, parents=True)

        f, ax = plt.subplots(4, 4, figsize=(16, 16))
        titles = ["Activity", "Resource"]
        k=0
        for i, dataset in enumerate(dataset_results.keys()):
            if i % 2 == 0:
                group = [ax[i-k][0], ax[i-k][1]]
            else:
                group = [ax[i-1-k][2], ax[i-1-k][3]]
                k = k + 1
            for j, subplot in enumerate(group):
                encoder=list(dataset_results[dataset].keys())[0]
                results = dataset_results[dataset][encoder]
                if j != 0:
                    metric = "Damerau-Levenshtein Resources"
                else:
                    metric = "Damerau-Levenshtein Acts"
                subplot.set_title(titles[j], fontsize=12)#, pad=15)
                subplot.set_xlabel('Prefix length')#, labelpad=15)
                subplot.set_ylabel(f'Avg. {metric}')#, labelpad=20)
                handles, labels = add_plot(subplot, dataset, metric, results, encoder)
                subplot.grid()
            dataset='Sepsis' if dataset == 'Sepsis_cases' else dataset
            group[0].set_title(dataset, fontsize=16, fontweight="bold", loc="center", pad=20)
        plt.tight_layout()#rect=[0, 0.15, 1, 0.6])
        f.subplots_adjust(top=0.96, bottom=0.11,hspace=0.4, wspace=0.3)
        # Create the legend
        legend = f.legend(labels=labels, title="Method:", bbox_to_anchor=(0.5, 0.01), loc="lower center",
                          ncol=3, borderaxespad=0., title_fontsize='large', fontsize='large')

        # Set the title font weight to bold
        legend.get_title().set_fontweight('bold')

        # Save the figure
        title = f"average_similarity_results_One_hot"
        plt.savefig(os.path.join(folder_path, f'{title}.pdf'))
        plt.close()

def prepare_data():
    dataset_results = {}
    prefix_result = []
    for dataset_id, dataset in enumerate(shared.log_list):
        # read evaluation data
        log_path = shared.log_folder / dataset
        evaluation_prefix_start = getprefix_start(log_path)
        df = pd.DataFrame()
        results_enc = {}
        for encoder in ["_One_hot"]:#,"_Simple_categorical", "_Combined_Act_res","_Multi_Enc"]:
            results = {}
            if encoder == "_One_hot":
                try:
                    data = aggregate_results(log_path,"SUTRAN", encoder)
                    results["SUTRAN"] = data
                except FileNotFoundError as not_found:
                    pass
            try:
                data = aggregate_results(log_path, "baseline",encoder, resource=True)
                results["SAP"] = data
            except FileNotFoundError as not_found:
                pass
            for i in [3]:#, 5,10
                try:
                    data = aggregate_results(log_path, "beamsearch", encoder, beam_size=i, resource=True)
                    x= "BS (bSize=" + str(i) + ")"
                    results[x] = data
                except FileNotFoundError as not_found:
                    pass
            for i in [3]:#, 5, 10
                try:
                    data = aggregate_results(log_path, "beamsearch", encoder, beam_size=i, resource=True, BK_end=True)
                    x = "BS + BK_END" + " (bSize=" + str(i) + ")"
                    results[x] = data
                except FileNotFoundError as not_found:
                    pass
            for i in [3]:#, 5, 10
                try:
                    data = aggregate_results(log_path, "beamsearch",encoder, beam_size=i, resource=True, BK=True)
                    x = "BS + BK" + " (bSize=" + str(i) + ")"
                    results[x] = data
                except FileNotFoundError as not_found:
                    pass
            if encoder == "_One_hot":
                encoders= "One-hot"
            elif encoder == "_Simple_categorical":
                encoders = "Index-based"
            elif encoder == "_Combined_Act_res":
                encoders = "shrinked Index-based"
            elif encoder =="_Multi_Enc":
                encoders = "Multi-Encoders"
            results_enc[encoders] = results
        dataset_results[(dataset.removesuffix('.xes')).capitalize() if dataset.removesuffix('.xes').islower() else dataset.removesuffix('.xes')] = results_enc
        prefix_result.append(list(range(evaluation_prefix_start, evaluation_prefix_start+ 5)))
    plot_results(dataset_results,prefix_result)

prepare_data()

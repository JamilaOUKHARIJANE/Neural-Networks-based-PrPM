from enum import Enum

import numpy as np
import pm4py

from src.commons import log_utils, shared_variables as shared
import csv
import os
import pandas as pd
from statistics import mean



class LogName(Enum):
    SEPSIS = 'Sepsis_cases'
    HELPDESK = 'helpdesk'
    BPIC11 = "BPI2011"
    BPIC12 = "BPI2012"
    BPIC13 = "BPI2013"
    BPIC13_In = "BPI2013_In"
    BPIC13_OP = "BPI2013_OP"
    BPIC13_CP = "BPI2013_CP"
    BPIC15_1 = "BPIC15_1"
    BPIC15_2 = "BPIC15_2"
    BPIC15_3 = "BPIC15_3"
    BPIC15_4 = "BPIC15_4"
    BPIC15_5 = "BPI2015_5"
    BPIC17 = "BPI2017"
    BPIC18 = "BPI2018"
    BPIC19 = "BPIC19"
    BPIC20_1 = "PrepaidTravelCost"
    BPIC20_2 = "PermitLog"
    BPIC20_3 = "RequestForPayment"
    BPIC20_4 = "DomesticDeclarations"
    BPIC20_5 = "InternationalDeclarations"

def set_log_ths(log_path):
    test_log = pm4py.read_xes(str(log_path).replace(".xes", f"_test.xes"))
    trace_ids = test_log['case:concept:name'].drop_duplicates().tolist()
    #trace_ids = None
    log_name = log_path.stem
    if log_name == LogName.HELPDESK.value:
        median = 5
        evaluation_prefix_start = median // 2 - 1
    elif log_name == LogName.BPIC11.value:
        median = 92
        evaluation_prefix_start = median // 2 - 2
    elif log_name == LogName.BPIC12.value:
        median = 32
        evaluation_prefix_start = median // 2 - 2
    elif log_name == LogName.BPIC13.value:
        median = 4
        evaluation_prefix_start = median // 2 - 1
    elif log_name == LogName.BPIC17.value:
        median = 54
        evaluation_prefix_start = median // 2 - 2
    elif log_name == LogName.SEPSIS.value:
        median = 13
        evaluation_prefix_start = median // 2 - 2
    elif log_name == LogName.BPIC15_1.value:
        median = 44
        evaluation_prefix_start = median // 2 - 2
    elif log_name == LogName.BPIC15_2.value:
        median = 54
        evaluation_prefix_start = median // 2 - 2
    elif log_name == LogName.BPIC15_3.value:
        median = 42
        evaluation_prefix_start = median // 2 - 2
    elif log_name == LogName.BPIC15_4.value:
        median = 44
        evaluation_prefix_start = median // 2 - 2
    elif log_name == LogName.BPIC15_5.value:
        median = 50
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
    elif log_name == LogName.BPIC20_5.value:
        median = 10
        evaluation_prefix_start = median // 2 - 2
    else:
        raise RuntimeError(f"No settings defined for log: {log_name.value}.")
    return evaluation_prefix_start, trace_ids




def aggregate_results(log_path, alg,models_folder,beam_size=3, resource=False,timestamp=False,outcome=False,probability_reduction=False, BK=False, BK_end=False, weight=0.0):
    average_act=[]
    average_res =[]
    average_length = []
    average_length_truth = []
    average_length_res = []
    for fold in range(shared.folds):
        eval_algorithm = alg + "_cf" + "r"*resource + "t"*timestamp + "o"*outcome
        log_name = log_path.stem
        if log_name in ['BPI2013','helpdesk'] and not BK_end:
            folder_path = shared.output_folder / models_folder / str(fold) / 'results' / eval_algorithm
        else:
            folder_path = shared.output_folder / models_folder / str(fold) / 'results_new' / eval_algorithm
        evaluation_prefix_start, test_trace_ids = set_log_ths(log_path)
        if alg == "beamsearch":
            filename = f'{log_name}_beam{str(beam_size)}_fold{str(fold)}_cluster{evaluation_prefix_start}{"_BKatEND" * BK_end}{"_mean_BK" * BK}.csv'
        else:
            filename = f'{log_name}_{str(alg)}_fold{str(fold)}_cluster{evaluation_prefix_start}.csv'
        file_path = os.path.join(folder_path, filename)
        if not os.path.exists(file_path):
            return 0, 0,0,0
        df_results = pd.read_csv(file_path, delimiter=',')
        df_results = df_results[df_results['Case ID'].isin(test_trace_ids)] if alg == "beamsearch" else df_results[df_results['Case Id'].isin(test_trace_ids)]
        if resource:
            if BK and log_name not in ['BPI2013','helpdesk']:
               df_results = df_results[df_results['Weight'] == weight]
            if log_name == 'helpdesk':
                df_results = df_results[df_results['Prefix length'] < 5]
            average_act.append(df_results['Damerau-Levenshtein Acts'].mean())
            average_res.append(df_results['Damerau-Levenshtein Resources'].mean())
            average_length.append(df_results['Predicted Acts'].str.len().mean())
            average_length_res.append(df_results['Predicted Resources'].str.len().mean())
            average_length_truth.append(df_results['Ground truth'].str.len().mean())
        else:
            average_act.append(df_results['Damerau-Levenshtein'].mean())
            average_length.append(df_results['Predicted'].str.len().mean())
            average_length_truth.append(df_results['Ground truth'].str.len().mean())
    if resource:
        print(f"{log_name}_{models_folder} - {eval_algorithm} -{BK * 'BK'}-{BK_end * 'BK_END'}", ":", round(mean(average_act), 3), ":",
              round(mean(average_res), 3), ":"
              , round(mean(average_length), 3), ":",         round(mean(average_length_truth), 3)
              )
        return round(mean(average_act), 3) , round(mean(average_res), 3), round(mean(average_length), 3) , round(mean(average_length_truth), 3)
    else:
        print(f"{log_name}_{models_folder} - {eval_algorithm}", ":", round(mean(average_act), 3), ":", round(mean(average_length), 3), ":",
              round(mean(average_length_truth), 3))
        return round(mean(average_act), 3), 0.0, round(mean(average_length), 3) , round(mean(average_length_truth), 3)


def getresults(log_list, algo, encoder, models_folder, beam_size=3 ,resource=False, BK=False, BK_end=False, weight=0.0):
    results = []
    for log in log_list:
        log_path = shared.log_folder / log
        average_act, average_res, _, _ = aggregate_results(log_path, algo, models_folder + encoder, beam_size=beam_size, resource=resource, BK=BK, BK_end= BK_end, weight=weight)
        results.append(average_act)
        if resource:
            results.append(average_res)
    return results

if __name__ == "__main__":
   encoders = ["_One_hot","_Simple_categorical", "_Combined_Act_res","_Multi_Enc"] #
   beam_sizes = [0,3,5,10]
   weights = [i / 10 for i in range(5, 10)]
   resource = True
   log_list = shared.log_list

   with (open(os.path.join(shared.output_folder, f"aggregated_results.csv"), mode='w') as out_file):
        writer = csv.writer(out_file, delimiter=',')
        headers = ["Method", "", "Encoder"]
        sub_headers = ["", "", ""]
        for log in log_list:
            headers.extend([log, ""]) if resource else headers.extend([log])
            sub_headers.extend(["Activities", "Resources"])  if resource else sub_headers.extend(["Activities"])
        headers.extend(['weight'])
        writer.writerow(headers)
        writer.writerow(sub_headers)
        for models_folder in ["keras_trans"]:
            for beam_size in beam_sizes:
                if beam_size == 0:
                    algo = 'baseline'
                    for encoder in encoders:
                        results = getresults(log_list,algo, encoder, models_folder, resource=resource)
                        writer.writerow([algo, beam_size, encoder.removeprefix("_")]+[res for res in results]+[0.0])
                else:
                    algo = 'beamsearch'
                    for encoder in encoders: #BS
                        results = getresults(log_list, algo,encoder,models_folder,beam_size=beam_size, resource=resource)
                        writer.writerow([algo,beam_size,encoder.removeprefix("_")]+[res for res in results]+[0.0])

                    for encoder in encoders:  # BS + BK end
                        results = getresults(log_list, algo, encoder, models_folder, beam_size=beam_size,
                                             resource=resource, BK=False, BK_end= True)
                        writer.writerow([algo + " + BK_END",beam_size, encoder.removeprefix("_")] + [res for res in results] + [0.0])

                    for encoder in encoders: #BS + BK
                        for weight in weights:
                            results = getresults(log_list, algo,encoder,models_folder,beam_size=beam_size, resource=resource, BK=True, BK_end=False, weight=weight)
                            writer.writerow([algo + " + BK",beam_size,encoder.removeprefix("_")]+[res for res in results]+[weight])
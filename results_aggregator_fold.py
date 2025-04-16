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

def set_log_ths(log_path,encoder=None, beam_size=0):
    test_log = pm4py.read_xes(str(log_path).replace(".xes", f"_test.xes"))
    trace_ids = test_log['case:concept:name'].drop_duplicates().tolist()
    log_name = log_path.stem
    weight =0
    if log_name == LogName.HELPDESK.value:
        median = 5
        evaluation_prefix_start = median // 2 - 1
    elif log_name == LogName.BPIC13.value:
        median = 4
        evaluation_prefix_start = median // 2 - 1
    elif log_name == LogName.SEPSIS.value:
        median = 13
        evaluation_prefix_start = median // 2 - 2
        if encoder == '_One_hot': weight=0.7
        elif encoder == '_Simple_categorical': weight=0.6
        elif encoder == '_Combined_Act_res':
            weight=0.7 if beam_size==3 else 0.6
        elif encoder == '_Multi_Enc':
            weight=0.9 if beam_size==5 else 0.8
    elif log_name == LogName.BPIC20_1.value:
        median = 8
        evaluation_prefix_start = median // 2 - 2
        if encoder == '_One_hot': weight = 0.7
        elif encoder == '_Simple_categorical':
            weight = 0.6 if beam_size == 10 else 0.8
        elif encoder == '_Combined_Act_res':
            weight = 0.7 if beam_size == 10 else 0.8
        elif encoder == '_Multi_Enc':
            weight = 0.8
    elif log_name == LogName.BPIC20_2.value:
        median = 11
        evaluation_prefix_start = median // 2 - 2
        if encoder == '_One_hot': weight=0.9
        elif encoder == '_Simple_categorical': weight=0.9
        elif encoder == '_Combined_Act_res':
            weight=0.8 if beam_size==10 else 0.6
        elif encoder == '_Multi_Enc':
            weight=0.9 if beam_size==3 else 0.7 if beam_size==5 else 0.5
    elif log_name == LogName.BPIC20_3.value:
        median = 5
        evaluation_prefix_start = median // 2 - 1
        if encoder == '_One_hot':
            weight = 0.8
        elif encoder == '_Simple_categorical':
            weight = 0.7
        elif encoder == '_Combined_Act_res':
            weight = 0.8
        elif encoder == '_Multi_Enc':
            weight = 0.8
    elif log_name == LogName.BPIC20_4.value:
        median = 5
        evaluation_prefix_start = median // 2 - 1
        weight = 0.9
    elif log_name == LogName.BPIC20_5.value:
        median = 10
        evaluation_prefix_start = median // 2 - 2
        if encoder == '_One_hot':
            weight = 0.7 if beam_size == 10 else 0.6
        elif encoder == '_Simple_categorical':
            weight = 0.6 if beam_size == 10 else 0.8
        elif encoder == '_Combined_Act_res':
            if beam_size == 10 : weight = 0.6
            elif beam_size==3: weight=0.8
            else: weight=0.7
        elif encoder == '_Multi_Enc':
            weight = 0.7
    else:
        raise RuntimeError(f"No settings defined for log: {log_name.value}.")
    return evaluation_prefix_start, trace_ids, weight


def aggregate_results(log_path, alg,models_folder,fold=0,beam_size=3, resource=False,timestamp=False,outcome=False, BK=False, BK_end=False, weight=0.0):
    average_act=[]
    average_res =[]
    average_combined = []
    log_name = log_path.stem
    if alg == "SUTRAN":
        folder_path = shared.output_folder / 'SUTRAN_results'
        filename = f'{log_name}_results.csv'
        file_path = os.path.join(folder_path, filename)
        if not os.path.exists(file_path):
            return 0, 0, 0
        df_results = pd.read_csv(file_path, delimiter=',')
        average_combined.append(df_results['dam_lev_similarity'].mean())
        average_combined.append(df_results['dam_lev_res_similarity'].mean())
        return round(df_results['dam_lev_similarity'].mean(), 3), round(df_results['dam_lev_res_similarity'].mean(), 3), round(mean(average_combined), 3)
    # Our methods
    evaluation_prefix_start, test_trace_ids, _ = set_log_ths(log_path, encoder, beam_size)
    for fold in range(fold, fold+1):
        eval_algorithm = alg + "_cf" + "r"*resource + "t"*timestamp + "o"*outcome
        if log_name in ['BPI2013','helpdesk'] and not BK_end: #
            folder_path = shared.output_folder / models_folder / str(fold) / 'results' / eval_algorithm
        else:
            folder_path = shared.output_folder / models_folder / str(fold) / 'results_new' / eval_algorithm
        if alg == "beamsearch":
            filename = f'{log_name}_beam{str(beam_size)}_fold{str(fold)}_cluster{evaluation_prefix_start}{"_BKatEND" * BK_end}{"_mean_BK" * BK}.csv'
        else: # baseline
            filename = f'{log_name}_{str(alg)}_fold{str(fold)}_cluster{evaluation_prefix_start}.csv'
        file_path = os.path.join(folder_path, filename)
        if not os.path.exists(file_path):
            return 0, 0,0
        df_results = pd.read_csv(file_path, delimiter=',')
        df_results = df_results[df_results['Case ID'].isin(test_trace_ids)] if alg == "beamsearch" else df_results[df_results['Case Id'].isin(test_trace_ids)]
        if BK and log_name not in ['BPI2013', 'helpdesk']:
            df_results = df_results[df_results['Weight'] == weight]
        if log_name == 'helpdesk':
            df_results = df_results[df_results['Prefix length'] < 5]
        average_act.append(df_results['Damerau-Levenshtein Acts'].mean())
        average_res.append(df_results['Damerau-Levenshtein Resources'].mean())
        average_combined.append(df_results['Damerau-Levenshtein Combined'].mean())
    return round(mean(average_act), 3) , round(mean(average_res), 3), round(mean(average_combined), 3)


def getresults(log_path, algo, encoder, models_folder="keras_trans", fold=0, beam_size=3 ,resource=False, BK=False, BK_end=False, weight=0.0, both=False):
    average_act, average_res, average_combined = aggregate_results(log_path, algo, models_folder + encoder,fold=fold, beam_size=beam_size, resource=True, BK=BK, BK_end= BK_end, weight=weight)
    if resource:
        return average_res
    elif both:
        return average_combined
    return average_act

if __name__ == "__main__":
   encoders = ["_One_hot","_Simple_categorical", "_Combined_Act_res","_Multi_Enc"]
   beam_sizes = [0,3,5,10]
   resource = True
   both = True
   log_list = shared.log_list
   BK_weight=True
   weights = [i / 10 for i in range(5, 10)]
   if BK_weight :
       with (open(os.path.join(shared.output_folder / 'Encodings_results',"aggregated_results_weights_new.csv"),
                  mode='w') as out_file):
           writer = csv.writer(out_file, delimiter=',')
           writer.writerow(["Dataset", "DLS_acts", "DLS_res", "DLS", "weight"])
           for log_name in shared.log_list:
               log_path = shared.log_folder / log_name
               for encoder in encoders:
                   for beam_size in beam_sizes:
                       if beam_size != 0:
                           for fold in range(shared.folds):
                               col1 = " " + log_name.removesuffix(".xes") + f"_{fold}{encoder}_{beam_size}"
                               if log_name in ['BPI2013.xes', 'helpdesk.xes']:
                                   average_act, average_res, average_combined = aggregate_results(log_path,
                                                                                                  "beamsearch",
                                                                                                  "keras_trans" + encoder,
                                                                                                  fold=fold,
                                                                                                  beam_size=beam_size,
                                                                                                  resource=resource,
                                                                                                  BK=True, BK_end=False)
                                   writer.writerow(
                                       [col1] + [average_act] + [average_res] + [average_combined] + [0.9])
                               else:
                                   for weight in weights:
                                       average_act, average_res, average_combined = aggregate_results(log_path, "beamsearch",
                                                                                                      "keras_trans"+encoder, fold=fold,
                                                                                                      beam_size=beam_size,
                                                                                                      resource=resource,
                                                                                                      BK=True, BK_end=False,
                                                                                                      weight=weight)
                                       writer.writerow([col1]+ [average_act]+ [average_res] + [average_combined]+ [weight])

   else:
       for encoder in encoders:
            if encoder == "_Simple_categorical":
                _encoder = "_Index_based"
            elif encoder == "_Combined_Act_res":
                _encoder = "_Product_Index_based"
            else:
                _encoder = encoder
            with (open(os.path.join(shared.output_folder /'Encodings_results', f"aggregated_results{_encoder}{'_Res' if resource else '_Acts'}{'_Res'*both}.csv"), mode='w') as out_file):
                writer = csv.writer(out_file, delimiter=',')
                headers = ["Dataset", "SAP", "BS (b=3)", "BS + BK_END (b=3)","BS + BK (b=3)"]
                if encoder == '_One_hot':
                    headers.extend(['SUTRAN'])
                writer.writerow(headers)
                for log_name in shared.log_list:
                    log_path = shared.log_folder / log_name
                    if encoder == '_One_hot':
                        algo = 'SUTRAN'
                        sutran_res = getresults(log_path, algo, encoder,resource=resource, both=both)
                    for fold in range(shared.folds):
                        results_fold = []
                        for beam_size in beam_sizes:
                            if beam_size == 0: # SAP
                                algo = 'baseline'
                                results_fold.append(getresults(log_path,algo, encoder,fold=fold, resource=resource, both=both))
                            else:
                                algo = 'beamsearch'
                                #BS
                                results_fold.append(getresults(log_path, algo,encoder,fold=fold, beam_size=beam_size, resource=resource, both=both))
                                # BS + BK end
                                results_fold.append(getresults(log_path, algo, encoder, fold=fold,  beam_size=beam_size,
                                                     resource=resource, BK=False, BK_end= True, both=both))
                                #BS + BK
                                results_fold.append(getresults(log_path, algo,encoder,fold=fold, beam_size=beam_size, resource=resource, BK=True, BK_end=False, both=both))
                        if encoder == '_One_hot':
                            writer.writerow([f" {log_name.removesuffix('.xes')}_{fold}"]+[res for res in results_fold]+[sutran_res])
                        else:
                            writer.writerow([f" {log_name.removesuffix('.xes')}_{fold}"] + [res for res in results_fold])
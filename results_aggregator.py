import numpy as np

from src.commons import log_utils, shared_variables as shared
import csv
import os
import pandas as pd
from statistics import mean


def aggregate_results(log_path, alg,models_folder,beam_size=3, resource=False,timestamp=False,outcome=False,probability_reduction=False, BK=False):
    average_act=[]
    average_res =[]
    average_length = []
    average_length_truth = []
    average_length_res = []
    for fold in range(2):#shared.folds):
        eval_algorithm = alg + "_cf" + "r"*resource + "t"*timestamp + "o"*outcome
        folder_path = shared.output_folder / models_folder / str(fold) / 'results' / eval_algorithm
        #print(f"fold {fold} - {eval_algorithm}")
        log_name = log_path.stem
        if log_name =='helpdesk':
            evaluation_prefix_start = 5 //2 - 1
        elif log_name == "BPI2011":
            evaluation_prefix_start =92 //2 - 2
        elif log_name =="BPI2013":
            evaluation_prefix_start =4 //2 - 1
        elif log_name == "BPI2012":
            evaluation_prefix_start = 32//2 - 2
        elif log_name =="BPI2017":
            evaluation_prefix_start = 54 //2 - 2
        if alg == "beamsearch":
            filename = f'{log_name}_beam{str(beam_size)}_fold{str(fold)}_cluster{evaluation_prefix_start}{"_probability_reduction" * probability_reduction}{"_mean_BK" * BK}.csv'
        else:
            filename = f'{log_name}_{str(alg)}_fold{str(fold)}_cluster{evaluation_prefix_start}.csv'

        file_path = os.path.join(folder_path, filename)
        if not os.path.exists(file_path):
            return 0, 0,0,0
        df_results = pd.read_csv(file_path, delimiter=',')
        average_act.append(df_results['Damerau-Levenshtein Acts'].mean())
        average_res.append(df_results['Damerau-Levenshtein Resources'].mean())
        average_length.append(df_results['Predicted Acts'].str.len().mean())
        average_length_res.append(df_results['Predicted Resources'].str.len().mean())
        average_length_truth.append(df_results['Ground truth'].str.len().mean())
        #print(df_results['Damerau-Levenshtein Acts'].mean(), df_results['Damerau-Levenshtein Resources'].mean())
        print(f"{log_name} : {df_results['Ground truth'].str.len().mean()}")
    print(f"{log_name}_{models_folder} - {eval_algorithm}", ":",round(mean(average_act), 3) , ":",round(mean(average_res), 3),":", round(mean(average_length), 3),":" , round(mean(average_length_truth), 3))
    return round(mean(average_act), 3) , round(mean(average_res), 3), round(mean(average_length), 3) , round(mean(average_length_truth), 3)

if __name__ == "__main__":
   with (open(os.path.join(shared.output_folder, f"aggregated_results.csv"), mode='w') as out_file):
        writer = csv.writer(out_file, delimiter=',')
        #writer.writerow(["", "","", "baseline", "", "beamsearch (beam_size=3)", "","","", "","","beamsearch (beam_size=5)", "","","","",""])
        writer.writerow(
            ["Method","", "Endoder",
             #"Helpdesk",  "","",
             "BPIC 2012",  "",#"",
             "BPIC 2013",  "",#"",
             "BPIC 2017", ""#,""
             ])
        writer.writerow(["",  "", "",
                         "Activities", "Resources",#"length",
                         "Activities", "Resources",#"length",
                         "Activities", "Resources",#"length",
                         #"Activities", "Resources","length"
                         ])
        for models_folder in ["keras_trans"]:
            for encoder in ["_One_hot","_Simple_categorical", "_Combined_Act_res","_Multi_Enc"]:
                results=[]
                for log in shared.log_list:
                    log_path = shared.log_folder / log
                    average_act_baseline, average_res_baseline, predictedlength, truth = aggregate_results(log_path, 'baseline',
                                                                                                           models_folder + encoder,resource=True,timestamp=False,outcome=False,probability_reduction=False)
                    results.append(average_act_baseline)
                    results.append(average_res_baseline)
                    results.append(predictedlength)
                writer.writerow(["baseline","" ,encoder.removeprefix("_"),
                                 results[0],results[1], #results[2],
                                 results[3], results[4], #results[5],
                                 results[6], results[7], #results[8],
                                 #results[9], results[10], results[11]
                                 ])

            for encoder in ["_One_hot","_Simple_categorical", "_Combined_Act_res","_Multi_Enc"]:
                results = []
                for log in shared.log_list:
                    log_path = shared.log_folder / log
                    average_act_beamsearch, average_res_beamsearch, predictedlength, _ = aggregate_results(
                            log_path, 'beamsearch',
                            models_folder + encoder, 3,
                            resource=True, timestamp=False,
                            outcome=False, probability_reduction=False, BK=False)
                    results.append(average_act_beamsearch)
                    results.append(average_res_beamsearch)
                    results.append(predictedlength)
                writer.writerow(["beamsearch", "bsize is 3", encoder.removeprefix("_"),
                                     results[0], results[1], #results[2],
                                     results[3], results[4], #results[5],
                                     results[6], results[7], #results[8],
                                     #results[9], results[10], results[11]
                                     ])
            for encoder in ["_One_hot","_Simple_categorical", "_Combined_Act_res","_Multi_Enc"]:
                results = []
                for log in shared.log_list:
                    log_path = shared.log_folder / log
                    average_act_beamsearch, average_res_beamsearch, predictedlength, _ = aggregate_results(
                            log_path, 'beamsearch',
                            models_folder + encoder, 5,
                            resource=True, timestamp=False,
                            outcome=False, probability_reduction=False, BK=False)
                    results.append(average_act_beamsearch)
                    results.append(average_res_beamsearch)
                    results.append(predictedlength)
                writer.writerow(["beamsearch", "bsize is 5", encoder.removeprefix("_"),
                                     results[0], results[1], #results[2],
                                     results[3], results[4], #results[5],
                                     results[6], results[7], #results[8],
                                     #results[9], results[10], results[11]
                                     ])
            '''for encoder in ["_One_hot","_Simple_categorical", "_Combined_Act_res","_Multi_Enc"]:
                results = []
                for log in shared.log_list:
                    log_path = shared.log_folder / log
                    average_act_beamsearch, average_res_beamsearch, predictedlength, _ = aggregate_results(
                            log_path, 'beamsearch',
                            models_folder + encoder, 10,
                            resource=True, timestamp=False,
                            outcome=False, probability_reduction=False, BK=False)
                    results.append(average_act_beamsearch)
                    results.append(average_res_beamsearch)
                    results.append(predictedlength)
                writer.writerow(["beamsearch", "bsize is 10", encoder.removeprefix("_"),
                                     results[0], results[1], results[2],
                                     results[3], results[4], results[5],
                                     results[6], results[7], results[8],
                                     #results[9], results[10], results[11]
                                     ])'''

            for encoder in ["_One_hot","_Simple_categorical","_Combined_Act_res", "_Multi_Enc"]: #
                results = []
                for log in shared.log_list:
                    log_path = shared.log_folder / log
                    average_act_beamsearch, average_res_beamsearch, predictedlength, _ = aggregate_results(
                            log_path, 'beamsearch',
                            models_folder + encoder, 3,
                            resource=True, timestamp=False,
                            outcome=False, probability_reduction=False, BK=True)
                    results.append(average_act_beamsearch)
                    results.append(average_res_beamsearch)
                    results.append(predictedlength)
                writer.writerow(["beamsearch with Declare BK", "bsize is 3", encoder.removeprefix("_"),
                                     results[0], results[1], #results[2],
                                     results[3], results[4], #results[5],
                                     results[6], results[7], #results[8],
                                     #results[9], results[10], results[11]
                                     ])
            for encoder in ["_One_hot","_Simple_categorical", "_Combined_Act_res","_Multi_Enc"]:
                results = []
                for log in shared.log_list:
                    log_path = shared.log_folder / log
                    average_act_beamsearch, average_res_beamsearch, predictedlength, _ = aggregate_results(
                            log_path, 'beamsearch',
                            models_folder + encoder, 5,
                            resource=True, timestamp=False,
                            outcome=False, probability_reduction=False, BK=True)
                    results.append(average_act_beamsearch)
                    results.append(average_res_beamsearch)
                    results.append(predictedlength)
                writer.writerow(["beamsearch with Declare BK", "bsize is 5", encoder.removeprefix("_"),
                                     results[0], results[1], #results[2],
                                     results[3], results[4], #results[5],
                                     results[6], results[7], #results[8],
                                     #results[9], results[10], results[11]
                                     ])
            '''for encoder in ["_One_hot","_Simple_categorical", "_Combined_Act_res","_Multi_Enc"]:
                results = []
                for log in shared.log_list:
                    log_path = shared.log_folder / log
                    average_act_beamsearch, average_res_beamsearch, predictedlength, _ = aggregate_results(
                            log_path, 'beamsearch',
                            models_folder + encoder, 10,
                            resource=True, timestamp=False,
                            outcome=False, probability_reduction=False, BK=True)
                    results.append(average_act_beamsearch)
                    results.append(average_res_beamsearch)
                    results.append(predictedlength)
                writer.writerow(["beamsearch with Declare BK", "bsize is 10", encoder.removeprefix("_"),
                                     results[0], results[1], results[2],
                                     results[3], results[4], results[5],
                                     results[6], results[7], results[8],
                                     #results[9], results[10], results[11]
                                     ])'''
import pdb
from src.commons import log_utils, shared_variables as shared
import csv
import os
import numpy as np
import pandas as pd
from statistics import mean


def aggregate_results(log_name, alg,models_folder,beam_size=3, resource=False,timestamp=False,outcome=False):
    average_act=[]
    average_res =[]
    for fold in range(shared.folds):
        eval_algorithm = alg + "_cf" + "r"*resource + "t"*timestamp + "o"*outcome
        folder_path = shared.output_folder / models_folder / str(fold) / 'results' / eval_algorithm
        print(f"fold {fold} - {eval_algorithm}")
        log_path = shared.log_folder / log_name
        log_data = log_utils.LogData(log_path)
        if alg == "beamsearch":
            filename = f'{log_data.log_name.value}_beam{str(beam_size)}_fold{str(fold)}_cluster{log_data.evaluation_prefix_start}.csv'
        else:
            filename = f'{log_data.log_name.value}_{str(alg)}_fold{str(fold)}_cluster{log_data.evaluation_prefix_start}.csv'

        df_results = pd.read_csv(os.path.join(folder_path, filename))
        average_act.append(df_results['Damerau-Levenshtein Acts'].mean())
        average_res.append(df_results['Damerau-Levenshtein Resources'].mean())
    print(models_folder, ":",mean(average_act), mean(average_res))
    return mean(average_act), mean(average_res)

if __name__ == "__main__":
    # Compute and save aggregated results
    with (open(os.path.join(shared.output_folder, f"aggregated_results_performance.csv"), mode='w') as out_file):
        writer = csv.writer(out_file, delimiter=',')
        writer.writerow(["", "","", "baseline", "", "beamsearch (beam_size=3)", "", "beamsearch (beam_size=5)", ""])
        writer.writerow(["Dataset", "Model", "Encoding", "Activities", "Resources", "Activities", "Resources", "Activities", "Resources"])
        for log in shared.log_list:
            for models_folder in ["keras_trans", "LSTM"]:
                for encoder in ["_One_hot", "_Simple_categorical", "_Combined_Act_res"]:
                    average_act_baseline, average_res_baseline = aggregate_results(log, 'baseline',models_folder + encoder,resource=True,timestamp=False,outcome=False)
                    average_act_beamsearch3, average_res_beamsearch3 = aggregate_results(log, 'beamsearch', models_folder + encoder, 3, resource=True, timestamp=False, outcome=False)
                    average_act_beamsearch5, average_res_beamsearch5 = aggregate_results(log, 'beamsearch',
                                                                                       models_folder + encoder, 5,
                                                                                       resource=True, timestamp=False,
                                                                                       outcome=False)
                    writer.writerow([log.removesuffix(".csv"), models_folder, encoder.removeprefix("_"),average_act_baseline,
                                 average_res_baseline, average_act_beamsearch3,average_res_beamsearch3, average_act_beamsearch5,average_res_beamsearch5])
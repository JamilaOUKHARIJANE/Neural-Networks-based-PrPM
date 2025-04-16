import os
import pickle

import pandas as pd
import torch


def load_dict(path_name):
    with open(path_name, 'rb') as file:
        loaded_dict = pickle.load(file)
    return loaded_dict

log_names= ['helpdesk','BPI2013','Sepsis_cases','PrepaidTravelCost','RequestForPayment','DomesticDeclarations','InternationalDeclarations', 'PermitLog']


for log_name in log_names:
    '''file = 'test_tensordataset.pt'
    res_file = os.path.join(log_name, file)
    res = torch.load(res_file)'''
    path = log_name +'/SUTRAN_NDA_results/TEST_SET_RESULTS'
    print("results for ", log_name)
    files_pkl = ['averaged_results.pkl', 'prefix_length_results_dict.pkl', 'suffix_length_results_dict.pkl']
    for file in files_pkl:
        res_file = os.path.join(path, file)
        results = load_dict(res_file)
        print(results)
    #'suffix_acts_decoded.pt','suffix_res_decoded.pt','labels.pt',
    pt_files = ['dam_lev_similarity.pt', 'dam_lev_res_similarity.pt', 'pref_len.pt', 'suf_len.pt'] #,'suffix_acts_decoded.pt', 'suffix_ttne_preds'
    df = pd.DataFrame()
    for file in pt_files:
        res_file = os.path.join(path, file)
        res = torch.load(res_file)
        #print(res)
        #df._append(pd.DataFrame(res.numpy()))
        df[file.removesuffix(".pt")] = pd.DataFrame(res.numpy())
    df.to_csv(path+f'/final_results.csv', index=False)
    if log_name == 'BPI2012':
        filtered_df = df[(df['pref_len'] >= 14) & (df['pref_len'] <= 18)]
    elif log_name == 'BPI2017':
        filtered_df = df[(df['pref_len'] >= 25) & (df['pref_len'] <= 29)]
    elif log_name =='BPI2013':
        filtered_df = df[(df['pref_len'] >= 1) & (df['pref_len'] <= 4)]
    elif log_name == 'helpdesk':
        filtered_df = df[(df['pref_len'] >= 1) & (df['pref_len'] <= 4)]
    elif log_name == 'Sepsis_cases':
        filtered_df = df[(df['pref_len'] >= 4) & (df['pref_len'] <= 8)]
    elif log_name == 'PrepaidTravelCost':
        filtered_df = df[(df['pref_len'] >= 2) & (df['pref_len'] <= 6)]
    elif log_name == 'DomesticDeclarations':
        filtered_df = df[(df['pref_len'] >= 1) & (df['pref_len'] <= 4)]
    elif log_name == 'InternationalDeclarations':
        filtered_df = df[(df['pref_len'] >= 3) & (df['pref_len'] <= 7)]
    elif log_name == 'RequestForPayment':
        filtered_df = df[(df['pref_len'] >= 1) & (df['pref_len'] <= 4)]
    elif log_name == 'PermitLog':
        filtered_df = df[(df['pref_len'] >= 3) & (df['pref_len'] <= 7)]
    filtered_df.to_csv(f'SUTRAN_results/{log_name}_results.csv', index=False)
    print("avg. act sim: ", round(filtered_df['dam_lev_similarity'].mean(),3))
    print("avg. res sim: ", round(filtered_df['dam_lev_res_similarity'].mean(),3))
    print("---------------------------------------------------")




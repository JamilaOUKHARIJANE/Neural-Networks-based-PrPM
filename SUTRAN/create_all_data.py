import pandas as pd 
import numpy as np
import pm4py

from Preprocessing.from_log_to_tensors import log_to_tensors
import os 
import torch

def preprocess(log):
    """Preprocess the event log.

    Parameters
    ----------
    log : pandas.DataFrame 
        Event log.

    Returns
    -------
    log : pandas.DataFrame
        Preprocessed event log.
    """
    # Convert timestamp column to datetime64[ns, UTC] dtype
    log['time:timestamp'] = pd.to_datetime(log['time:timestamp']).dt.tz_convert('UTC') #tz_localize('UTC')
    return log


def construct_datasets(log_name):
    event_log = pm4py.read_xes(f"logs/{log_name}.xes")
    event_log = event_log.dropna(subset=['org:resource'])
    df = pm4py.convert_to_dataframe(event_log)
    df = preprocess(df)
    categorical_casefeatures = []
    numeric_eventfeatures = []
    categorical_eventfeatures = []
    numeric_casefeatures = []
    case_id = 'case:concept:name'
    timestamp = 'time:timestamp'
    act_label = 'concept:name'
    res_label = 'org:resource'

    if log_name == 'helpdesk':
        window_size = 15
    elif log_name == 'BPI2012':
        window_size = 168
    elif log_name == 'BPI2013':
        window_size = 123
    elif log_name == 'BPI2017':
        window_size = 180
    elif log_name == 'Sepsis_cases':
        window_size = 61
    elif log_name == 'DomesticDeclarations':
        window_size = 24
    elif log_name == 'InternationalDeclarations':
        window_size = 27
    elif log_name == 'PrepaidTravelCost':
        window_size = 21
    elif log_name == 'RequestForPayment':
        window_size = 20
    elif log_name == 'PermitLog':
        window_size = 90

    start_date = None
    end_date = None
    max_days = 1343.33
    start_before_date = "2018-09"
    test_len_share = 0.2
    val_len_share = 0.2
    mode = 'simple'
    outcome = None
    result = log_to_tensors(df, 
                            log_name=log_name, 
                            start_date=start_date, 
                            start_before_date=start_before_date,
                            end_date=end_date, 
                            max_days=max_days, 
                            test_len_share=test_len_share, 
                            val_len_share=val_len_share, 
                            window_size=window_size, 
                            mode=mode,
                            case_id=case_id, 
                            act_label=act_label,
                            res_label= res_label,
                            timestamp=timestamp, 
                            cat_casefts=categorical_casefeatures, 
                            num_casefts=numeric_casefeatures,
                            cat_eventfts=categorical_eventfeatures, 
                            num_eventfts=numeric_eventfeatures, 
                            outcome=outcome)
    
    train_data, val_data, test_data = result

    # Create the log_name subfolder in the root directory of the repository
    # (Should already be created when having executed the `log_to_tensors()`
    # function.)
    output_directory = log_name
    os.makedirs(output_directory, exist_ok=True)

    # Save training tuples
    train_tensors_path = os.path.join(output_directory, 'train_tensordataset.pt')
    torch.save(train_data, train_tensors_path)

    # Save validation tuples
    val_tensors_path = os.path.join(output_directory, 'val_tensordataset.pt')
    torch.save(val_data, val_tensors_path)

    # Save test tuples
    test_tensors_path = os.path.join(output_directory, 'test_tensordataset.pt')
    torch.save(test_data, test_tensors_path)

logs= ['PermitLog','helpdesk','PrepaidTravelCost','RequestForPayment','DomesticDeclarations','InternationalDeclarations']
for log_name in logs:
    construct_datasets(log_name)
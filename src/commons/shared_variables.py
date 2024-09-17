"""
This file was created in order to bring
common variables and functions into one file to make
code more clear
"""
from pathlib import Path


ascii_offset = 161
beam_size = 3 #3
th_reduction_factor = 1
usePosEncoding = False
useProb_reduction = False

root_folder = Path.cwd() #/ 'implementation_real_logs'
data_folder = root_folder / 'data'
input_folder = data_folder / 'input'
output_folder = data_folder / 'output'

log_folder = input_folder / 'logs'
pn_folder = input_folder / 'petrinets'

epochs = 100
folds = 3
train_ratio = 0.9
variant_split = 0.9
validation_split = 0.2

log_list = [
    #'sepsis_cases_1.csv',
    #'helpdesk.csv',
    # 'BPIC12.csv' ,
    # 'Road_Traffic.csv',
    # 'BPIC13_I.csv',
     #'BPIC13_CP.csv',
    # 'Synthetic.xes',
    #'Production.csv',
    #'10x20_3S.csv',
    #'10x20_3W.csv',
    '10x5_1S.csv',
    #'50x5_1S.csv'
]

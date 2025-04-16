"""
This file was created in order to bring
common variables and functions into one file to make
code more clear
"""
from pathlib import Path

version = "_mean"
ascii_offset = 161
beam_size = 3
th_reduction_factor = 1
One_hot_encoding=False
combined_Act_res = False
useProb_reduction = False
use_modulator = False
use_train_test_logs = True
declare_BK = False
BK_end = False
root_folder = Path.cwd() #/ 'implementation_real_logs'
data_folder = root_folder / 'data'
input_folder = data_folder / 'input'
output_folder = data_folder / 'output'

declare_folder = input_folder / 'declare_models'
xes_log_folder =  input_folder / 'log_xes'
log_folder = input_folder / 'logs'
pn_folder = input_folder / 'petrinets'

epochs = 100
folds = 3
train_ratio = 0.8
variant_split = 0.9
validation_split = 0.2
iteration = 0

log_list = [
    'helpdesk.xes',
   'Sepsis_cases.xes',
   'BPI2013.xes',
    'DomesticDeclarations.xes',
   'InternationalDeclarations.xes',
    'PrepaidTravelCost.xes',
   'RequestForPayment.xes',
     'PermitLog.xes',


]

